import warnings

warnings.filterwarnings('ignore')  # setting ignore as a parameter
import os
import re
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import argparse
import contextlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
from datetime import timedelta, datetime
from torch.utils.data import DataLoader, Dataset
from nn.utils import chunk_and_normalize
import torchaudio
from fairseq import checkpoint_utils
from scipy import interpolate
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normalize",
        default=True,
        type=str2bool,
        help="Should we normalize the input to zero mean and unit variance",
    )
    parser.add_argument(
        "--write-non-focal",
        default=True,
        type=str2bool,
        help="If True then we also output non-focal predictions.",
    )
    parser.add_argument(
        "--write-embeddings",
        default=False,
        type=str2bool,
        help="If True then we also output the "
             "frame-wise embeddings of the transformer network."
             "If target information is also provided, they will be written"
             "to the embedding file as well."
             "This will take up a LOT of disk space."
             "The embeddings of 1h from a 8kHz file will use 2.2GB.",
    )
    parser.add_argument(
        "--average_top_k_layers", default=12,
        type=int,
        help="The transformer layer with which we end averaging"
    )
    parser.add_argument(
        "--visualize",
        default=False,
        type=str2bool,
        help="If True then we do a plot of the preds.",
    )
    parser.add_argument(
        "--only-highest-likelihood",
        default=True,
        type=str2bool,
        help="If True then we export only these predictions with the highest likelihoods"
             "in case of multiple overlapping predictions at the same time.",
    )
    parser.add_argument(
        "--write-other-predictions",
        default=True,
        type=str2bool,
        help="This is only used when '--only-highest-likelihood' is enabled."
             "If True then we export two prediction files. One that holds only the"
             "predictions with the highest likelihood and a second one that"
             "holds all other predictions."
    )
    parser.add_argument(
        "--overwrite-previous-preds",
        default=False,
        type=str2bool,
        help="If True then we overwrite a previous prediction if the write-out filename "
             "matches. If False (default), then we skip these files.",
    )
    parser.add_argument(
        "--method",
        default="avg",
        type=str,
        choices=["avg", "max", "canny"],
        help="Which method to use for fusing the predictions into time bins."
             "avg: Average pooling, then thresholding."
             "max: Max pooling, then thresholding."
             "canny: Canny edge detector",
    )
    parser.add_argument(
        "--sigma-s",
        default=0.1,
        type=float,
        help="Size of Gaussian (std dev) in seconds for the canny method. "
             "Filter width in seconds for avg and max methods.",
    )
    parser.add_argument(
        "--metric-threshold",
        default=0.15,
        type=float,
        help="Threshold for the filtered predictions. Only used when --method={max, avg}",
    )
    parser.add_argument(
        "--iou-threshold",
        default=0.,
        type=float,
        help="The minimum IoU that is needed for a prediction to be counted as "
             "overlapping with focal. 0 means that even a single frame overlap is enough, "
             "while 1 means that only a perfect overlap is counted as overlap.",
    )
    parser.add_argument(
        "--maxfilt-s",
        default=0.1,
        type=float,
        help="Time to smooth with maxixmum filter. Only used when --method=canny",
    )
    parser.add_argument(
        "--max-duration-s",
        default=0.5,
        type=float,
        help="Detections are never longer than max duration (s). Only used when --method=canny",
    )
    parser.add_argument(
        "--lowP",
        default=0.125,
        type=float,
        help="Low threshold. Detections are based on Canny edge detector, "
             "but the detector sometimes misses the minima on some transients"
             "if they are not sharp enough.  lowP is used for pruning detections."
             "Detections whose Gaussian-smoothed signal is beneath lowP are discarded."
             "Only used when --method=canny",
    )
    parser.add_argument(
        "--sample-rate",
        default=8000,
        type=int,
        help="Sample rate to which everything is resampled. Please note that the models are"
             "currently trained for input at 8kHz.",
    )
    parser.add_argument(
        "--path", default="",
        type=str,
        help="Files for which predictions are created. Can be a path to a file or a folder"
    )
    parser.add_argument(
        "--target-path", default="",
        type=str,
        help="(Optional) File of folder for which targets are available. "
             "This is only used when --visualize=True"
             "The same numpy structure as during training is assumed."
    )
    parser.add_argument(
        "--channel-info", default="",
        type=str,
        help="[Optional] Path to a csv file that holds information "
             "about which channel to use when a file is stereo."
             "Alternatively a number can be passed."
             "If nothing is passed then channel 0 is chosen."
    )
    parser.add_argument(
        "--model-path",
        default="checkpoint_last.pt",
        type=str,
        help="Path to pretrained model. "
             "This should point to a *.pt file created by pytorch."
    )
    parser.add_argument(
        "--device", default="cuda",
        type=str, choices=["cuda", "cpu"],
        help="The device you want to use. "
             "We will fall back to cpu if no cuda device is available."
    )
    parser.add_argument(
        "--batch-size", default=16,
        type=int,
        help="Every file that is being split into smaller segments and fed to the model"
             "in batches. --batch-size gives this batch size and --segment-length gives"
             "the segment length in sec. Default values are 1s segment length and "
             "256 segments batch size. This is optimized for a GPU with 12GB memory."
             "Adjust if you have more - or less. Keep in mind that first reducing --batch-size"
             "is recommend as reducing segment-length reduces the timespan the transformer"
             "can use for contextualizing."
    )
    parser.add_argument(
        "--segment-length", default=10.,
        type=float,
        help="Every file that is being split into smaller segments and fed to the model"
             "in batches. --batch-size gives this batch size and --segment-length gives"
             "the segment length in sec. Default values are 1s segment length and "
             "256 segments batch size. This is optimized for a GPU with 12GB memory."
             "Adjust if you have more - or less. Keep in mind that first reducing --batch-size"
             "is recommend as reducing segment-length reduces the timespan the transformer"
             "can use for contextualizing."
    )
    parser.add_argument(
        "--min-pred-length", default=.05,
        type=float,
        help="Every prediction that has a length shorter than this (in s) is discarded"
    )
    parser.add_argument(
        "--out-path", default=os.path.curdir,
        type=str,
        help="The path to where the tab separated csv files should be written to."
             "This has to be a folder name, not a filename. The output filenames are "
             "fixed to correspond to the input wav filenames. The folder does not need"
             "to exists. The script can take care of that. Default is the current workdir."
    )
    parser.add_argument(
        "--unique-values",
        default="['beep', 'synch', 'sn', 'cc', 'ld', 'oth', 'mo', 'al', 'soc', 'agg', 'eating', 'focal']",
        type=str,
        help="A string list that, when evaluated, holds the names for the used classes."
    )
    return parser


def hz_to_mel(frequencies, *, htk=False):
    """Convert Hz to Mels

    frequencies : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk : bool
        use HTK formula instead of Slaney
    Returns
    -------
    mels : number or np.ndarray [shape=(n,)]
        input frequencies in Mels
    See Also
    --------
    mel_to_hz
    """

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, *, htk=False):
    """Convert mel bin numbers to frequencies

    mels : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk : bool
        use HTK formula instead of Slaney
    Returns
    -------
    frequencies : np.ndarray [shape=(n,)]
        input mels in Hz
    See Also
    --------
    hz_to_mel
    """

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def mel_frequencies(n_mels=128, *, fmin=0.0, fmax=11025.0, htk=False):
    """Compute an array of acoustic frequencies tuned to the mel scale.
    The mel scale is a quasi-logarithmic function of acoustic frequency
    designed such that perceptually similar pitch intervals (e.g. octaves)
    appear equal in width over the full hearing range.
    Because the definition of the mel scale is conditioned by a finite number
    of subjective psychoaoustical experiments, several implementations coexist
    in the audio signal processing literature [#]_. By default, librosa replicates
    the behavior of the well-established MATLAB Auditory Toolbox of Slaney [#]_.
    According to this default implementation,  the conversion from Hertz to mel is
    linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
    replicates the Hidden Markov Toolkit [#]_ (HTK) according to the following formula::
        mel = 2595.0 * np.log10(1.0 + f / 700.0).
    The choice of implementation is determined by the ``htk`` keyword argument: setting
    ``htk=False`` leads to the Auditory toolbox implementation, whereas setting it ``htk=True``
    leads to the HTK implementation.
    .. [#] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
        In Proc. International Conference on Acoustics, Speech, and Signal Processing
        (ICASSP), vol. 1, pp. 217-220, 1998.
    .. [#] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
        Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.
    .. [#] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
        Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
        The HTK book, version 3.4. Cambridge University, March 2009.
    n_mels : int > 0 [scalar]
        Number of mel bins.
    fmin : float >= 0 [scalar]
        Minimum frequency (Hz).
    fmax : float >= 0 [scalar]
        Maximum frequency (Hz).
    htk : bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.
    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of ``n_mels`` frequencies in Hz which are uniformly spaced on the Mel
        axis.
    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


def get_mel_spectrum(input):
    """
    Convert audio signal to Log-Mel-Energy Spectrogram

    input: the input audio file
    """

    # The following parameters are based on:
    # Thomas, M., et al. (2022)
    # A practical guide for generating unsupervised, spectrogram-based
    # latent space representations of animal vocalizations
    # The Journal of Animal Ecology. https://doi.org/10.1111/1365-2656.13754

    n_mels = 40  # --> number of mel bins
    fft_win = 0.03  # --> length of audio chunk when applying stft in seconds
    fft_hop = fft_win / 8  # --> hop_length in seconds

    mel_bands = mel_frequencies(n_mels=n_mels, fmax=int(args.sample_rate / 2))
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=int(fft_win * args.sample_rate),
        win_length=int(fft_win * args.sample_rate),
        hop_length=int(fft_hop * args.sample_rate),
        n_mels=n_mels,
        f_max=int(args.sample_rate / 2),
        norm="slaney",
        mel_scale="slaney"
    )
    mel_power = torchaudio.transforms.AmplitudeToDB(stype="power")  # to dB scale
    # Do MelSpectrogram, convert to numpy, and return
    mel = mel_spectrogram(input)
    mel = mel_power(mel)
    return mel.squeeze().detach().cpu().numpy(), mel_bands


def visualize(probs, fused_preds, segment_length, save_path,
              wav, multiplier, targets=None):
    """
    A helper function for visualizing the fused predictions along
    with ground truth information if available.
    This is slowing down everything considerably, as we not only calculate
    a Mel Spectrogram but also interpolate the time, probs, preds, and targets linearly
    in order to have a shared x-axis. But it looks nice and is useful for getting an overview.

    probs: The model output probabilities
    fused_preds: The output of the fuse_predict routine
    segment_length: Length of the given snippet. For building the time vector
    save_path: The output filename. Directories along the way will be created.
    wav: The original wav input (used only for Mel Spectrogram)
    multiplier: since one batch of len n may not be enough to do inference for the - possibly very long - file
                we introduce a multiplier parameter, with which the time and idx axis can be shifted.
    targets: (Optional) If ground truth is available, it will be plotted

    """
    modifier = probs.size(0) * (0 + multiplier)
    unique_labels = eval(args.unique_values)
    main_fs = 14
    with plt.rc_context({
        'font.size': main_fs,
    }):
        f, ax = plt.subplots(nrows=len(unique_labels) + 1, ncols=1,
                             figsize=(int(1.6 * segment_length),
                                      int(len(unique_labels) * 5)),
                             sharex="all"
                             )
        mel, mel_bands = get_mel_spectrum(wav)
        ax[0].imshow(mel, origin="lower")
        for ci, ul in enumerate(unique_labels):  # we iterate through all classes and do a plot for each
            _probs = probs[:, ci]
            time_start = args.segment_length * args.batch_size * (0 + multiplier)
            time_end = time_start + segment_length
            time = torch.linspace(time_start,
                                  time_end,
                                  len(_probs))
            # For having a shared x-axis, we interpolate the time, probs, preds, and targets linearly
            f_prob = interpolate.interp1d(time, _probs, kind="linear")
            interp_time = torch.linspace(time_start,
                                         time_end,
                                         mel.shape[1])
            interp_probs = f_prob(interp_time)
            # We use the time vector for actual plotting and later change the ticklabels, as the
            # share_x by matplotlib uses index based alignment.
            time_vector = torch.arange(interp_time.shape[0])

            ax[ci + 1].plot(time_vector, interp_probs, lw=2,
                            label="Prediction by model",
                            alpha=1)
            ax[ci + 1].set_ylim((0, 1))

            preds = torch.ones_like(time)
            for ii, gt in enumerate(fused_preds[1]):
                if len(gt[ci]) > 0:
                    for gtt in gt[ci]:
                        preds[gtt[0] - modifier:gtt[1] - modifier + 1] = .51  # Plot it only until the half of the axes
            f_tar = interpolate.interp1d(time, preds, kind="linear")
            interp_preds = f_tar(interp_time)
            ax[ci + 1].fill_between(
                x=time_vector,
                y1=torch.ones_like(time_vector) * 1.,
                y2=interp_preds,
                color="tab:blue",
                label="Fused predictions",
                alpha=.25
            )
            if targets is not None:
                f_tar = interpolate.interp1d(time, targets[:, ci], kind="linear")
                interp_targets = f_tar(interp_time)
                ax[ci + 1].fill_between(
                    x=time_vector,
                    y1=torch.zeros_like(time_vector),
                    y2=interp_targets * .49,
                    color="tab:green",
                    label="Ground truth",
                    alpha=.25
                )

        title_str_args = (args.method, args.sigma_s * 1000, args.metric_threshold)
        title_str = "Fused predictions using the '{}' method with a window " \
                    "length of {:02.0f}ms and a threshold of {:02.02f}".format(*title_str_args)
        ax[0].set_title(title_str)
        ax[-1].set_xlabel("Time in seconds")
        x_ticks = time_vector[::int(len(time_vector) / segment_length * 2)]  # Every 2s a tick
        ax[-1].set_xticks(x_ticks)
        ax[-1].set_xticklabels(["{:01.0f}".format(interp_time[x]) for x in x_ticks])
        for ii, a in enumerate(ax):
            if ii > 0:
                a.set_ylabel("Model likelihood")
                leg = a.legend(loc=3, framealpha=1, edgecolor="w",
                               title="{}".format(unique_labels[ii - 1]),
                               title_fontsize=1.5 * main_fs)
                leg._legend_box.align = "left"
                a.grid(True, axis="x")
            else:  # The Spectrogram
                a.grid(False)
                a.set_xlim(left=0)
                a.set_ylabel("Mel bins in Hz")
                y_ticks = np.linspace(0, len(mel) - 1, 5).round().astype(int)
                a.set_yticks(y_ticks)
                a.set_yticklabels(["{:01.0f}".format(x) for x in mel_bands[y_ticks]])
                a.set_aspect("auto")
            sns.despine(top=True, right=True, left=True, bottom=True,
                        offset=25, trim=True, ax=a)
        f.tight_layout()
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        f.savefig(os.path.join(save_path),
                  dpi=72)
        plt.close(f)


def export_embeddings(model, layer_results):
    avg_layer = args.average_top_k_layers

    finetuned = False
    if hasattr(model, "w2v_encoder"):  # is finetuned or not
        finetuned = True

    if finetuned:
        target_layer_results = [l[0] for l in layer_results[-avg_layer:]]
    else:
        target_layer_results = [l for l in layer_results[-avg_layer:]]

    # Average over transformer layers
    avg_emb = (sum(target_layer_results) / len(target_layer_results)).float()  # BTC
    # print("\n\nexport:\n")
    # print(avg_emb.shape)

    return avg_emb.detach().cpu().numpy()


def prepare_data_and_embeddings(net_output, samples, batch, unique_labels, multiplier):
    """
    We prepare the logits and targets for the visualization and/or embedding write-out routine

    net_output : The output of the model
    samples: The samples provided by the DataLoader
    batch: The processed batch by this' script main routine
    unique_labels: The unique labels
    multiplier: since one batch of len n may not be enough to do inference for the - possibly very long - file
                we introduce a multiplier parameter, with which the time and idx axis can be shifted.
    """
    probs = torch.sigmoid(net_output["encoder_out"].clone().detach().cpu()).float()
    # we build a modifier that shift that labels in case we have multiple batches for a single file
    modifier = args.segment_length * args.sample_rate * args.batch_size * (0 + multiplier)
    # we concatenate the batch and the probs for visualization

    concat_probs = torch.concat(torch.unbind(probs))  # BTC -> TC
    concat_batch = torch.concat(torch.unbind(batch))  # BT -> T
    # TODO: When this is at the end of a batched file, then timing information is screwed up for the
    # last batch, as the number of segments may be smaller in that batch
    segment_length = concat_batch.size(0) / args.sample_rate
    target_vector = None
    if os.path.isdir(args.target_path):
        do_target = True
        npz_file = os.path.join(
            args.target_path, os.path.basename(samples["filename"][0]).replace(".wav", ".npz")
        )
        # You can create this pickle file with the 'raw_labels_to_dataframe.py' script
        pickle_file = os.path.join(args.target_path, "meerkat_label_file.P")
        if os.path.isfile(npz_file):
            fname = os.path.join(
                args.target_path,
                os.path.basename(samples["filename"][0]).replace(".wav", ".h5")
            )
            with h5py.File(fname, "r") as f:
                # We shift everything according to multiplier
                start_frame_lbl = [x - modifier for x in list(f["start_frame_lbl"])]
                end_frame_lbl = [x - modifier for x in list(f["end_frame_lbl"])]
                lbl_cat = list(f["lbl_cat"])
                foc = list(f["foc"])  # TODO: Allow inference when focal is absent
        elif os.path.isfile(pickle_file):
            df = pd.read_pickle(pickle_file)
            df = df.where(df.realFile == os.path.basename(
                samples["filename"][0])).dropna(how="all")
            if len(df) > 0:
                start_frame_lbl = list(np.squeeze(df.StartRelative.to_frame().applymap(
                    lambda x: round(x.total_seconds() * args.sample_rate) - modifier
                ).values))
                end_frame_lbl = list(np.squeeze(df.EndRelative.to_frame().applymap(
                    lambda x: round(x.total_seconds() * args.sample_rate) - modifier
                ).values))
                lbl_cat = list(np.squeeze(df.Name.to_frame().applymap(
                    lambda x: unique_labels.index(x)
                ).values))
                foc = list(np.squeeze(df.Focal.to_frame().applymap(
                    lambda x: 1 if x.lower() == "focal" else 0
                ).values))
            else:
                do_target = False
        else:
            do_target = False

        if do_target:
            source_vector = np.zeros((concat_batch.size(0), len(unique_labels)), dtype=np.int64)
            source_idx_vector = np.arange(concat_batch.size(0))
            target_vector_idx_vector = np.round(np.linspace(0, concat_batch.size(0),
                                                            concat_probs.size(0),
                                                            endpoint=False)).astype(np.int64)
            if len(start_frame_lbl) > 0 and len(end_frame_lbl) > 0 and len(lbl_cat) > 0:
                for ii, (s, e, l) in enumerate(zip(start_frame_lbl, end_frame_lbl, lbl_cat)):
                    if s > 0 and e < len(source_vector):
                        source_vector[int(s):int(e), int(l)] = 1
                        # is it a focal call or not
                        focal_lbl = foc[ii]
                        source_vector[int(s):int(e), -1] = focal_lbl

            f = interpolate.interp1d(source_idx_vector, source_vector,
                                     axis=0, kind="linear")
            target_vector = f(target_vector_idx_vector).astype(np.int64)
        else:

            print(
                "\nWe couldn't find target information for: {} ... "
                "skipping target visualization".format(os.path.basename(samples["filename"][0]))
            )

    return {
        "probs": concat_probs,
        "segment_length": segment_length,
        "wav": concat_batch,
        "targets": target_vector,
    }


def replace_ending(st, target_ending=".csv"):
    st_ = st
    supported_file_types = ["WAV", "AIFF", "AIFC", "FLAC", "OGG", "MP3", "MAT"]
    for ft in supported_file_types:
        regex_ft = "." + "".join(["[{}{}]".format(x.lower(), x.upper()) for x in ft])
        st_ = re.sub(regex_ft, "{}".format(target_ending), st_)
    return st_


def to_td_from_sec(s):
    """helper function for creating timedelta from seconds"""
    return timedelta(seconds=float(s))


def flatten(l):
    """helper function for flattening a list of lists"""
    return [item for sublist in l for item in sublist]


def sort_multiple(lists):
    """helper function for sorting multiple lists simultaneously"""
    return zip(*sorted(zip(*lists)))


def overlaps_iou(start_1, duration_1, start_2, duration_2):
    """
    Get temporal overlap based on two timedeltas (start and duration) for two events
    """
    # 1: is the start of the current call earlier than the start of some other
    #    prediction and is this start of the other prediction earlier than the
    #    end of the current call.
    #    -> Current call prediction overlaps the start of some other prediction
    cond1 = start_1 <= start_2 < (start_1 + duration_1)
    # 2: is the start of the current call earlier than the end of some other prediction
    #    and is the end of the current call later
    #    -> Current call prediction overlaps the end of some other prediction
    cond2 = start_1 < (start_2 + duration_2) <= (start_1 + duration_1)
    # 3: is the start of some other call earlier than the start of the current call
    #    and is the end of the other call later than the end of the current call
    #   -> Other call prediction overlaps the full length of the current call
    #   (!) Please note that the only left scenario (Current call prediction fully
    #       overlaps other call prediction is covered by 1 and 2)
    cond3 = start_2 < start_1 < (start_1 + duration_1) < (start_2 + duration_2)

    overlap_condition = cond1 or cond2 or cond3

    if overlap_condition:
        interval1_end = start_1 + duration_1
        interval2_end = start_2 + duration_2

        # Calculate start and end times of intersection
        intersection_start = max(start_1, start_2)
        intersection_end = min(interval1_end, interval2_end)

        # Calculate duration of intersection and union
        intersection_duration = max(timedelta(), intersection_end - intersection_start)
        union_duration = duration_1 + duration_2 - intersection_duration

        # Calculate IoU
        if union_duration != 0:
            iou = intersection_duration / union_duration
        else:
            iou = 0.
    else:
        iou = 0.

    return overlap_condition, iou


class FileFolderDataset(Dataset):
    def __init__(self, path, sample_rate=8000, channel_info=None):
        self.path = path
        self.sample_rate = sample_rate

        # Check if path is a file or folder
        if os.path.isfile(self.path):
            self.fnames_all = [self.path]
        elif os.path.isdir(self.path):
            self.fnames_all = list(Path(self.path).rglob('*.wav')) + list(Path(self.path).rglob('*.WAV'))
            self.fnames_all = [str(x) for x in self.fnames_all]  # Convert from PosixPath to string
        else:
            raise NotImplementedError("The given path ({}) is not a path nor a directory.".format(self.path))

        self.fnames = []
        for ff in self.fnames_all:
            # Conditions before we go on
            cond1 = "sound_calibration" not in ff.lower()  # No Calibration files
            cond2 = "old" not in ff.lower()  # Nothing that was tagged "old"
            cond3 = os.path.basename(ff)[0] != "."  # No empty hidden files that mysteriously are *.wav files
            if cond1 is True and cond2 is True and cond3 is True:
                try:
                    _ = sf.info(ff)  # soundfile.info to see if the file is corrupt
                    self.fnames.append(ff)  # move to the next row in the dataframe
                except sf.LibsndfileError as e:
                    print("{} is corrupt and raised a LibsndfileError".format(ff), flush=True)
        assert len(self.fnames) > 0  # Make sure we still have some wav files

        # Read in channel info
        self.channel_info = channel_info
        if channel_info is not None and os.path.isfile(channel_info):
            channel_tab = pd.read_csv(channel_info, sep=",")
            self.channel_info = {x.lower(): y for x, y in zip(channel_tab.wavFile, channel_tab.meerkatChannel)}
        elif channel_info is not None and channel_info.isdigit():
            self.channel_info = int(channel_info)

    def __len__(self):
        return len(self.fnames)

    def postprocess(self, feats, curr_sample_rate, file_name=None):

        if feats.dim() == 2:
            channel_idx = feats.abs().sum(0).argmax()  # We choose the channel, with higher amplitude
            if self.channel_info is not None:
                if isinstance(self.channel_info, dict):
                    if file_name.lower() in self.channel_info and file_name is not None:
                        channel_idx = self.channel_info[os.path.basename(file_name).lower()]
                elif isinstance(self.channel_info, int):
                    channel_idx = self.channel_info
            feats = feats[:, channel_idx]

        if curr_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(  # Same as librosa "Kaiser Best", but faster
                curr_sample_rate,
                self.sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
                dtype=feats.dtype,
            )
            try:
                feats = resampler(feats.view(1, -1))
            except Exception as e:
                print(
                    "\n\nException raised during resampling\n\n",
                    file_name,
                    feats.size(),
                    curr_sample_rate,
                    self.sample_rate
                )
                raise e

        feats = feats.squeeze()
        assert feats.dim() == 1, feats.dim()
        return feats

    def __getitem__(self, index):

        try:
            wav, curr_sample_rate = sf.read(self.fnames[index], dtype="float32")
        except sf.LibsndfileError as e:
            print("{} is corrupt and raised a LibsndfileError. We skip it.".format(self.fnames[index]), flush=True)
            return int(0)
        if len(wav) == 0:
            print("{} is corrupt and has a length of zero. We skip it.".format(self.fnames[index]), flush=True)
            return int(0)

        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate, self.fnames[index])

        return {"id": index, "source": feats,
                "filename": self.fnames[index]}


def main(args):
    # TODO: Multi-GPU inference
    # TODO: Individual window length for each class
    # TODO: Individual metric thresholds for each class
    # fused predictions dict, see argparse documentation
    method_dict = {
        "sigma_s": args.sigma_s,
        "metric_threshold": args.metric_threshold,
        "maxfilt_s": args.maxfilt_s,
        "max_duration_s": args.max_duration_s,
        "lowP": args.lowP,
    }
    print("Passed arguments:\n", args)
    # The labels on which the model was trained on
    unique_labels = eval(args.unique_values)
    # Create the dataset
    dataset = FileFolderDataset(args.path, sample_rate=args.sample_rate,
                                channel_info=args.channel_info)
    # create the dataloader
    dataloader = DataLoader(dataset,
                            batch_size=1,  # One file at a time
                            shuffle=False)

    model_path = args.model_path
    assert os.path.isfile(model_path)

    # Check if out path exists
    if not os.path.exists(os.path.join(args.out_path, "csv")):
        os.makedirs(os.path.join(args.out_path, "csv"))

    # load the model
    print("\n Loading model ...", end="")
    models, _model_args = checkpoint_utils.load_model_ensemble([model_path])
    print("done")
    device = "cuda" if torch.cuda.is_available() and args.device.lower() == "cuda" else "cpu"
    print("Moving model to {} ...".format(device), end="")
    model = models[0].to(device)  # place on appropriate device
    print("done\n")

    # print("\n\n", _model_args, "\n\n")

    # The header for every csv file
    header = ["Name", "Start", "Duration", "Time Format", "Type", "Description"]

    with torch.inference_mode():
        model.eval()
        for samples in tqdm(dataloader, total=len(dataset.fnames),
                            desc="Iterating over {} input file(s)".format(len(dataset.fnames))):
            if not isinstance(samples, dict):  # dataloader returned 0 as the current file is corrupt
                if samples.dim():  # double check that the data loader returned a scalar
                    continue
            current_file = samples["filename"][0]
            fname = replace_ending(
                os.path.basename(current_file),
                target_ending=".csv"
            )
            if args.write_embeddings:
                file_args = (os.path.basename(current_file), model.__class__.__name__,
                             os.path.basename(model_path),
                             args.average_top_k_layers)
                emb_out_filename = "{}_embeddings_{}_{}_{}.h5".format(*file_args)
                out_path = os.path.join(args.out_path, emb_out_filename)
                if os.path.isfile(out_path):
                    os.remove(out_path)
                embedding_context = h5py.File(out_path, "w")
            else:
                embedding_context = contextlib.nullcontext()

            # if the only-highest-likelihood mode is enabled but the other predictions are also
            # wanted, the write-other-predictions mode can be enabled. This will write out a
            # second file that holds all the secondary predictions
            write_secondary_pred_file = args.only_highest_likelihood and args.write_other_predictions
            file_out = os.path.join(os.path.join(args.out_path, "csv"), fname)
            file_exists = os.path.isfile(file_out)
            # write to a secondary file when write_secondary_pred_file is True
            if write_secondary_pred_file:
                fname_sec = replace_ending(
                    os.path.basename(current_file),
                    target_ending="_secondary_predictions.csv"
                )
                file_out_secondary = os.path.join(os.path.join(args.out_path, "csv"), fname_sec)
                secondary_file_exists = os.path.isfile(file_out_secondary)

            if not args.overwrite_previous_preds:
                # if we should've written two prediction files, check for both
                if write_secondary_pred_file and file_exists and secondary_file_exists:
                    continue
                elif not write_secondary_pred_file and file_exists:
                    continue

            # we prepare lists that hold the intervals detected by the fuse_predict routine
            time_intervals = []
            likelihoods = []
            # get the input file and check if it is too long
            wav = samples["source"].squeeze()
            batched_wav = chunk_and_normalize(
                wav,
                segment_length=args.segment_length,
                sample_rate=args.sample_rate,
                normalize=False,
                max_batch_size=args.batch_size)
            tqdm_args = (len(batched_wav), len(batched_wav[0]), args.segment_length)
            with embedding_context as f_h5:
                if args.write_embeddings:
                    f_h5.create_dataset(name="filename", data=current_file)
                embeddings = []
                for bi, batch in enumerate(tqdm(batched_wav, desc="Inference for {} batch(es) of {} "
                                                                  "sequences each of length {}s".format(*tqdm_args),
                                                leave=False)):
                    if not torch.is_tensor(batch):
                        batch = torch.stack(batch)  # stack the list of tensors
                    elif batch.dim() == 1:  # split segments or single segment
                        batch = batch.view(1, -1)
                    if args.normalize:
                        batch = torch.stack([torch.nn.functional.layer_norm(x, x.shape) for x in batch])
                    # get some predictions
                    net_output = model(source=batch.to(device))

                    if args.write_embeddings:
                        layer_results = model.extract_features(source=batch.to(device))["layer_results"]
                        layer_results = [dd.reshape(dd.size(0) * dd.size(1), dd.size(2)) for dd in layer_results]
                        # print("\n\nembeddings\n", len(layer_results), layer_results[0].shape, "\n\n")
                        embeddings.append(export_embeddings(model, layer_results))

                    if "linear_eval_projection" in net_output:
                        probs = torch.sigmoid(net_output["linear_eval_projection"].float())
                    elif "encoder_out" in net_output:
                        probs = torch.sigmoid(net_output["encoder_out"].float())
                    else:
                        print("No logits returned. It probably is a pretrained model without finetuning")
                        continue
                    fused_preds = model.fuse_predict(batch.size(-1), probs,
                                                     method_dict=method_dict,
                                                     method=args.method,
                                                     multiplier=bi,
                                                     bs=args.batch_size)
                    # Should we visualize
                    if args.visualize or args.write_embeddings:
                        f_name_args = [
                            os.path.basename(current_file).replace(".wav", "").replace(".WAV", ""),
                            bi
                        ]
                        kwargs = prepare_data_and_embeddings(net_output, samples, batch, unique_labels, bi)
                        if args.visualize:
                            save_path = os.path.join(args.out_path, "gfx",
                                                     "{}_{}.pdf".format(*f_name_args))
                            visualize(fused_preds=fused_preds,
                                      save_path=save_path,
                                      multiplier=bi, **kwargs)
                        if args.write_embeddings:
                            print("yes")

                    _time_interval = fused_preds[0]
                    _likelihoods = fused_preds[2]
                    # both interval list are nested such that the first dim is sequence-batches (batch),
                    # then class, then a tuple with start end information
                    if len(_time_interval) > 0:
                        time_intervals.append(_time_interval)
                        likelihoods.append(_likelihoods)

                if args.write_embeddings:  # Iterate over Batch
                    emb_data = np.concatenate(embeddings)
                    f_h5.create_dataset(name="embedding", data=emb_data, dtype=np.float32)  # TC
#                    o_len_mult = np.ceil(wav.shape[0] / args.sample_rate / args.segment_length).astype(int)
#                    o_time_vec = np.linspace(
#                        start=0,
#                        stop=o_len_mult * args.segment_length,
#                        num=int(o_len_mult * args.segment_length * args.sample_rate)
#                    )
#                    interp_time_vec = np.interp(
#                        x=np.linspace(
#                            start=0,
#                            stop=len(o_time_vec),
#                            num=int(emb_data.shape[0])
#                        ),
#                        xp=np.arange(len(o_time_vec)),
#                        fp=o_time_vec
#                    )
#                    f_h5.create_dataset(name="time", data=interp_time_vec, dtype=np.float32)  # TC
#
                    true_dur = wav.shape[0] / args.sample_rate   # duration in seconds of the (resampled) wav tensor
                    T = emb_data.shape[0]
                    time_vec = np.linspace(0.0, true_dur, num=T, endpoint=False, dtype=np.float32)
                    f_h5.create_dataset(name="time", data=time_vec, dtype=np.float32)
                    
            # the write out routine
            time_intervals = flatten(time_intervals)
            likelihoods = flatten(likelihoods)
            file_context_one = open(file_out, "w")
            file_context_two = open(file_out_secondary, "w") if write_secondary_pred_file else contextlib.nullcontext()
            with file_context_one as f, file_context_two as f_sec:  # open file for writing
                f.write("\t".join(header) + "\n")  # write header -> tab separated
                write_out_lines = {}
                if write_secondary_pred_file:
                    write_out_lines_secondary = {}
                    f_sec.write("\t".join(header) + "\n")  # write header -> tab separated
                # we now iterate over the batches
                for ti, lik in tqdm(zip(time_intervals, likelihoods),
                                    desc="Fusing and writing the found signals to file",
                                    leave=False):
                    # Check that we have individual predictions for all classes. No One-Hot
                    assert net_output["encoder_out"].dim() == 3
                    assert len(ti) == len(unique_labels)
                    # we exclude the last prediction, as this is the focal predictor
                    starts = flatten([[to_td_from_sec(x[0]) for x in y] for y in ti[:-1]])
                    durations = flatten([[to_td_from_sec(x[1] - x[0]) for x in y] for y in ti[:-1]])
                    names = flatten([[unique_labels[ii] for _ in y] for ii, y in enumerate(ti[:-1])])
                    likes = flatten(lik[:-1])

                    if len(starts) == 0 or len(durations) == 0 or len(names) == 0 or len(likes) == 0:
                        continue  # don't bother if we didn't found anything and continue

                    starts, durations, names, likes = sort_multiple([starts, durations, names, likes])
                    focal_starts = [to_td_from_sec(x[0]) for x in ti[-1]]
                    focal_durations = [to_td_from_sec(x[1] - x[0]) for x in ti[-1]]

                    # first we iterate through all predicted calls and try to assign focal / non-focal labels
                    # to each call. We save every write-out line in a dictionary. If at the end focal
                    # predictions are left for which no call prediction could be assigned, then we write
                    # these predictions as "unk" in the prediction file.
                    for event_idx, (sa, du, n, li) in enumerate(zip(starts, durations, names, likes)):

                        # We need to check if there are overlapping predictions with a higher likelihood.
                        # If so, then only assign the focal label to the one with the highest likelihood and, further,
                        # if the only_highest_likelihood mode is activated, then discard all other predictions at the
                        # given time, except for the one with the highest model likelihood. or
                        # write to a secondary file when write_secondary_pred_file is True
                        # we calculate this by first iterating backwards though the starts, and duration list
                        # and look for overlaps and then forward though both lists.
                        highest_likelihood = li
                        # Going backward
                        lesser_overlaps_names = []
                        lesser_overlaps_starts = []
                        lesser_overlaps_ends = []
                        lesser_overlaps_likelihoods = []
                        if event_idx > 0:
                            for back_idx, (s_, d_, n_, l_) in enumerate(zip(
                                    starts[:event_idx][::-1],
                                    durations[:event_idx][::-1],
                                    names[:event_idx][::-1],
                                    likes[:event_idx][::-1]
                            ),
                            ):
                                overlaps, iou = overlaps_iou(sa, du, s_, d_)
                                new_likelihood = likes[event_idx - back_idx - 1]
                                if overlaps and iou > args.iou_threshold and new_likelihood > highest_likelihood:
                                    # we found a likelihood that is higher,
                                    # which means we will write out a prediction for this time segment
                                    # at another predictions. Therefore, we break here and move forward
                                    # with sa, du, n, and li
                                    highest_likelihood = new_likelihood
                                    break
                                elif not overlaps:
                                    break
                                # print("adding back")
                                lesser_overlaps_names.append(n_)
                                lesser_overlaps_starts.append(s_.total_seconds())
                                lesser_overlaps_ends.append(s_.total_seconds() + d_.total_seconds())
                                lesser_overlaps_likelihoods.append(l_)

                        # Going forward only if no higher likelihood has been found
                        if event_idx < (len(starts) - 1) and highest_likelihood == li:
                            for forw_idx, (s_, d_, n_, l_) in enumerate(
                                    zip(
                                        starts[event_idx + 1:],
                                        durations[event_idx + 1:],
                                        names[event_idx + 1:],
                                        likes[event_idx + 1:]
                                    )
                            ):
                                overlaps, iou = overlaps_iou(sa, du, s_, d_)
                                new_likelihood = likes[event_idx + forw_idx + 1]
                                if overlaps and iou > args.iou_threshold and new_likelihood > highest_likelihood:
                                    highest_likelihood = new_likelihood  # we found a likelihood that is higher -> break
                                    break
                                elif not overlaps:
                                    break
                                # print("adding forw")
                                lesser_overlaps_names.append(n_)
                                lesser_overlaps_starts.append(s_.total_seconds())
                                lesser_overlaps_ends.append(s_.total_seconds() + d_.total_seconds())
                                lesser_overlaps_likelihoods.append(l_)

                        is_highest_likelihood = highest_likelihood == li

                        # We iterate through all focal predictions to see if some call was focal or not
                        foc = 0
                        for fi, (fs, fd) in enumerate(zip(focal_starts, focal_durations)):
                            overlaps, iou = overlaps_iou(sa, du, fs, fd)
                            if overlaps and iou > args.iou_threshold and is_highest_likelihood:
                                foc = 1

                                # We remove this prediction so it cannot be
                                # assigned to another call prediction
                                focal_starts.pop(fi)
                                focal_durations.pop(fi)
                                # we end the iteration here as a single call
                                # can only be predicted focal a single time
                                break
                        # write-out only with disabled only_highest_likelihood mode or
                        # when is_highest_likelihood and enabled only_highest_likelihood mode or
                        # write to a secondary file when write_secondary_pred_file is True
                        likelihood_write_out = ((args.only_highest_likelihood and is_highest_likelihood)
                                                or not args.only_highest_likelihood)

                        desc = "{:1.03f}".format(li)
                        if args.only_highest_likelihood and is_highest_likelihood and len(
                                lesser_overlaps_likelihoods) > 0:
                            sorted_overlaps = list(sort_multiple(
                                [
                                    lesser_overlaps_likelihoods,
                                    lesser_overlaps_ends,
                                    lesser_overlaps_starts,
                                    lesser_overlaps_names
                                ]
                            ))
                            sorted_overlaps = [x[:3] for x in sorted_overlaps]  # take top three
                            desc += " | Alternative predictions: "
                            base_str = []
                            for l in zip(*sorted_overlaps):
                                base_str.append(
                                    "{} from {:1.02f} to {:1.02f}s with likelihood {:1.02f}".format(*l[::-1]))
                            desc += ", ".join(base_str)

                        if du > to_td_from_sec(args.min_pred_length) and (
                                likelihood_write_out or write_secondary_pred_file):
                            # save only when larger than minimum length and when focal
                            # name = n + " nf" if foc == 0 and n not in ["beep", "synch"] else n
                            if foc == 1:
                                # Here, we don't need to check if it is a primary or secondary prediction
                                # as the foc flag already checked it has highest likelihood
                                read_out_line = [n, str(sa), str(du), "decimal", "Cue", desc]
                                write_out_lines.update({str(sa) + "_" + n: read_out_line})

                            # if non-focal should also be written out
                            if args.write_non_focal and foc == 0:
                                if write_secondary_pred_file and not is_highest_likelihood:
                                    # Only write to secondary file when the flag is True and when
                                    # it is not the highest likelihood. The prediction can be the one
                                    # with the highest likelihood and not being focal as these are
                                    # independent classification heads, so we need to double check this here.
                                    read_out_line = [n, str(sa), str(du), "decimal", "Cue", desc]
                                    write_out_lines_secondary.update({str(sa) + "_" + n: read_out_line})
                                else:
                                    read_out_line = [n + " nf", str(sa), str(du), "decimal", "Cue", desc]
                                    write_out_lines.update({str(sa) + "_" + n: read_out_line})

                    if len(focal_starts) > 0:  # We have unassigned focal calls left
                        for fs_unk, fd_unk in zip(focal_starts, focal_durations):
                            if fd_unk > to_td_from_sec(args.min_pred_length):  # only when longer than minimum time
                                read_out_line = ["unk", str(fs_unk), str(fd_unk), "decimal", "Cue", ""]
                                write_out_lines.update({str(fs_unk) + "_unk": read_out_line})

                # sort the dictionary and finally write out to file
                for wl in sorted(write_out_lines.items()):
                    f.write("\t".join(wl[1]) + "\n")
                if write_secondary_pred_file:
                    for wl in sorted(write_out_lines_secondary.items()):
                        f_sec.write("\t".join(wl[1]) + "\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
