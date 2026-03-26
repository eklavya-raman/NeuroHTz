import mne


def make_epochs(raw: mne.io.Raw, event_id: dict[str, int], tmin: float, tmax: float) -> mne.Epochs:
    """Create epochs using the standard stimulation channel."""
    events = mne.find_events(raw, stim_channel="STI 014")
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None)
    return epochs


def segment_epochs(epochs: mne.Epochs, segment_length: float) -> mne.Epochs:
    """Segment each epoch into fixed-length windows and concatenate them."""
    segmented_epochs = []
    for epoch in epochs:
        n_segments = int((epoch.times[-1] - epoch.times[0]) / segment_length)
        for i in range(n_segments):
            start_time = epoch.times[0] + i * segment_length
            end_time = start_time + segment_length
            segmented_epoch = epoch.copy().crop(tmin=start_time, tmax=end_time)
            segmented_epochs.append(segmented_epoch)

    if not segmented_epochs:
        raise ValueError("No segmented epochs were created; check segment_length and epoch duration.")
    return mne.concatenate_epochs(segmented_epochs)
import mne

# Make epochs from raw data
def make_epochs(raw: mne.io.Raw, event_id: dict[str, int], tmin: float, tmax: float) -> mne.Epochs:
    events = mne.find_events(raw, stim_channel='STI 014')
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None)
    return epochs


# Segmentation of epochs into fixed-length segments
def segment_epochs(epochs: mne.Epochs, segment_length: float) -> mne.Epochs:
    segmented_epochs = []
    for epoch in epochs:
        n_segments = int((epoch.times[-1] - epoch.times[0]) / segment_length)
        for i in range(n_segments):
            start_time = epoch.times[0] + i * segment_length
            end_time = start_time + segment_length
            segmented_epoch = epoch.copy().crop(tmin=start_time, tmax=end_time)
            segmented_epochs.append(segmented_epoch)
    return mne.concatenate_epochs(segmented_epochs)

