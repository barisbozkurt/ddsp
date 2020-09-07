# Copyright 2020 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Apache Beam pipeline for computing TFRecord dataset from audio files."""

from absl import logging
import apache_beam as beam
from ddsp import spectral_ops
import numpy as np
import pydub
import tensorflow.compat.v2 as tf



def _load_audio_as_array(audio_path: str,
                         sample_rate: int) -> np.array:
  """Load audio file at specified sample rate and return an array.

  When `sample_rate` > original SR of audio file, Pydub may miss samples when
  reading file defined in `audio_path`. Must manually zero-pad missing samples.

  Args:
    audio_path: path to audio file
    sample_rate: desired sample rate (can be different from original SR)

  Returns:
    audio: audio in np.float32
  """
  with tf.io.gfile.GFile(audio_path, 'rb') as f:
    # Load audio at original SR
    audio_segment = (pydub.AudioSegment.from_file(f).set_channels(1))
    # Compute expected length at given `sample_rate`
    expected_len = int(audio_segment.duration_seconds * sample_rate)
    # Resample to `sample_rate`
    audio_segment = audio_segment.set_frame_rate(sample_rate)
    audio = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    # Zero pad missing samples, if any
    audio = spectral_ops.pad_or_trim_to_expected_length(audio, expected_len)
  # Convert from int to float representation.
  audio /= 2**(8 * audio_segment.sample_width)
  return audio


def _load_audio(audio_path, sample_rate):
  """Load audio file."""
  logging.info("Loading '%s'.", audio_path)
  beam.metrics.Metrics.counter('prepare-tfrecord', 'load-audio').inc()
  audio = _load_audio_as_array(audio_path, sample_rate)
  return {'audio': audio}


def add_loudness(ex, sample_rate, frame_rate, n_fft=2048):
  """Add loudness in dB."""
  beam.metrics.Metrics.counter('prepare-tfrecord', 'compute-loudness').inc()
  audio = ex['audio']
  mean_loudness_db = spectral_ops.compute_loudness(audio, sample_rate,
                                                   frame_rate, n_fft)
  ex = dict(ex)
  ex['loudness_db'] = mean_loudness_db.astype(np.float32)
  return ex

def add_lsfs(ex, sample_rate, frame_rate):
  """Add line spectral frequencies"""
  beam.metrics.Metrics.counter('prepare-tfrecord', 'compute-lsfs').inc()
  audio = ex['audio']
  n_lsfs = 20
  lsfs = spectral_ops.compute_lsfs(audio, 
                                   expected_len = ex['loudness_db'].size,
                                   n_lsfs=n_lsfs,
                                   sample_rate=sample_rate, 
                                   frame_size=512)
  ex = dict(ex)
  for i in range(n_lsfs):
      ex['lsfs_'+str(i)] = lsfs[:,i].astype(np.float32)
  
  #ex['lsfs'] = np.mean(lsfs, axis=1)
  print('audio.shape', ex['audio'].shape,
        '\nloudness_db.shape', ex['loudness_db'].shape,
        '\nf0_hz.shape', ex['f0_hz'].shape,
        '\nf0_confidence.shape', ex['f0_confidence'].shape,
        '\nlsfs_0.shape', ex['lsfs_0'].shape,
        '\nlsfs_1.shape', ex['lsfs_1'].shape)
  
  return ex

def _add_f0_estimate(ex, sample_rate, frame_rate):
  """Add fundamental frequency (f0) estimate using CREPE."""
  beam.metrics.Metrics.counter('prepare-tfrecord', 'estimate-f0').inc()
  audio = ex['audio']
  f0_hz, f0_confidence = spectral_ops.compute_f0(audio, sample_rate, frame_rate)
  ex = dict(ex)
  ex.update({
      'f0_hz': f0_hz.astype(np.float32),
      'f0_confidence': f0_confidence.astype(np.float32)
  })
  return ex


def split_example(
    ex, sample_rate, frame_rate, window_secs, hop_secs):
  """Splits example into windows, padding final window if needed."""

  def get_windows(sequence, rate):
    window_size = int(window_secs * rate)
    hop_size = int(hop_secs * rate)
    n_windows = int(np.ceil((len(sequence) - window_size) / hop_size))  + 1
    n_samples_padded = (n_windows - 1) * hop_size + window_size
    n_padding = n_samples_padded - len(sequence)
    sequence = np.pad(sequence, (0, n_padding), mode='constant')
    for window_end in range(window_size, len(sequence) + 1, hop_size):
      yield sequence[window_end-window_size:window_end]

  for audio, loudness_db, f0_hz, f0_confidence, lsfs_0, lsfs_1, lsfs_2, lsfs_3, lsfs_4, lsfs_5, lsfs_6, lsfs_7, lsfs_8, lsfs_9, lsfs_10, lsfs_11, lsfs_12, lsfs_13, lsfs_14, lsfs_15, lsfs_16, lsfs_17, lsfs_18, lsfs_19 in zip(
      get_windows(ex['audio'], sample_rate),
      get_windows(ex['loudness_db'], frame_rate),
      get_windows(ex['f0_hz'], frame_rate),
      get_windows(ex['f0_confidence'], frame_rate),
      get_windows(ex['lsfs_0'], frame_rate),
      get_windows(ex['lsfs_1'], frame_rate),
      get_windows(ex['lsfs_2'], frame_rate),
      get_windows(ex['lsfs_3'], frame_rate),
      get_windows(ex['lsfs_4'], frame_rate),
      get_windows(ex['lsfs_5'], frame_rate),
      get_windows(ex['lsfs_6'], frame_rate),
      get_windows(ex['lsfs_7'], frame_rate),
      get_windows(ex['lsfs_8'], frame_rate),
      get_windows(ex['lsfs_9'], frame_rate),
      get_windows(ex['lsfs_10'], frame_rate),
      get_windows(ex['lsfs_11'], frame_rate),
      get_windows(ex['lsfs_12'], frame_rate),
      get_windows(ex['lsfs_13'], frame_rate),
      get_windows(ex['lsfs_14'], frame_rate),
      get_windows(ex['lsfs_15'], frame_rate),
      get_windows(ex['lsfs_16'], frame_rate),
      get_windows(ex['lsfs_17'], frame_rate),
      get_windows(ex['lsfs_18'], frame_rate),
      get_windows(ex['lsfs_19'], frame_rate)):
    beam.metrics.Metrics.counter('prepare-tfrecord', 'split-example').inc()
    yield {
        'audio': audio,
        'loudness_db': loudness_db,
        'f0_hz': f0_hz,
        'f0_confidence': f0_confidence,
        'lsfs_0': lsfs_0,
        'lsfs_1': lsfs_1,
        'lsfs_2': lsfs_2,
        'lsfs_3': lsfs_3,
        'lsfs_4': lsfs_4,
        'lsfs_5': lsfs_5,
        'lsfs_6': lsfs_6,
        'lsfs_7': lsfs_7,
        'lsfs_8': lsfs_8,
        'lsfs_9': lsfs_9,
        'lsfs_10': lsfs_10,
        'lsfs_11': lsfs_11,
        'lsfs_12': lsfs_12,
        'lsfs_13': lsfs_13,
        'lsfs_14': lsfs_14,
        'lsfs_15': lsfs_15,
        'lsfs_16': lsfs_16,
        'lsfs_17': lsfs_17,
        'lsfs_18': lsfs_18,
        'lsfs_19': lsfs_19
    }


def float_dict_to_tfexample(float_dict):
  """Convert dictionary of float arrays to tf.train.Example proto."""
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              k: tf.train.Feature(float_list=tf.train.FloatList(value=v))
              for k, v in float_dict.items()
          }
      ))


def prepare_tfrecord(
    input_audio_paths,
    output_tfrecord_path,
    num_shards=None,
    sample_rate=16000,
    frame_rate=250,
    window_secs=4,
    hop_secs=1,
    pipeline_options=''):
  """Prepares a TFRecord for use in training, evaluation, and prediction.

  Args:
    input_audio_paths: An iterable of paths to audio files to include in
      TFRecord.
    output_tfrecord_path: The prefix path to the output TFRecord. Shard numbers
      will be added to actual path(s).
    num_shards: The number of shards to use for the TFRecord. If None, this
      number will be determined automatically.
    sample_rate: The sample rate to use for the audio.
    frame_rate: The frame rate to use for f0 and loudness features.
      If set to None, these features will not be computed.
    window_secs: The size of the sliding window (in seconds) to use to
      split the audio and features. If 0, they will not be split.
    hop_secs: The number of seconds to hop when computing the sliding
      windows.
    pipeline_options: An iterable of command line arguments to be used as
      options for the Beam Pipeline.
  """
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      pipeline_options)
  with beam.Pipeline(options=pipeline_options) as pipeline:
    examples = (
        pipeline
        | beam.Create(input_audio_paths)
        | beam.Map(_load_audio, sample_rate))

    if frame_rate:
      examples = (
          examples
          | beam.Map(_add_f0_estimate, sample_rate, frame_rate)
          | beam.Map(add_loudness, sample_rate, frame_rate)
          | beam.Map(add_lsfs, sample_rate, frame_rate))

    if window_secs:
      examples |= beam.FlatMap(
          split_example, sample_rate, frame_rate, window_secs, hop_secs)

    _ = (
        examples
        | beam.Reshuffle()
        | beam.Map(float_dict_to_tfexample)
        | beam.io.tfrecordio.WriteToTFRecord(
            output_tfrecord_path,
            num_shards=num_shards,
            coder=beam.coders.ProtoCoder(tf.train.Example))
    )
