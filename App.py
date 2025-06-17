import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython import display

# Install required libraries
try:
    import soundfile
except ImportError:
    !pip install -q tensorflow==2.11.* librosa soundfile pandas matplotlib
    import soundfile

# Set random seed
tf.random.set_seed(42)
np.random.seed(42)

# Load YAMNet model
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
try:
    yamnet_model = hub.load(yamnet_model_handle)
    print("YAMNet model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load YAMNet model: {e}")

# Utility function to load and preprocess WAV files
def load_wav_16k_mono(filename_tensor):
    """Load a WAV file, resample to 16 kHz mono, and convert to a float tensor."""
    try:
        filename_str = filename_tensor.numpy().decode('utf-8')
        wav, sr = librosa.load(filename_str, sr=16000, mono=True)
        min_samples = int(0.96 * 16000)  # YAMNet requires 0.96 seconds
        if len(wav) < min_samples:
            wav = np.pad(wav, (0, max(0, min_samples - len(wav))), mode='constant')
        max_val = np.max(np.abs(wav)) if np.max(np.abs(wav)) > 0 else 1.0
        wav = wav / max_val
        wav = tf.convert_to_tensor(wav, dtype=tf.float32)
        wav = tf.clip_by_value(wav, -1.0, 1.0)
        return wav
    except Exception as e:
        print(f"Error loading {filename_str}: {e}")
        return tf.zeros([int(0.96 * 16000)], dtype=tf.float32)

# Instrument classes and mapping
my_classes = ['drums', 'guitar']
map_class_to_id = {instrument: idx for idx, instrument in enumerate(my_classes)}

# Infer instrument from filename
def infer_instrument_from_filename(filename):
    filename_lower = filename.lower()
    if 'drum' in filename_lower:
        return 'drums'
    elif 'guitar' in filename_lower:
        return 'guitar'
    raise ValueError(f"Filename {filename} does not contain 'drum' or 'guitar'.")

# Load WAV files
data_path = './new_audio'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Directory {data_path} does not exist.")
filenames = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.lower().endswith('.wav')]
if not filenames:
    raise ValueError(f"No .wav files found in {data_path}.")

# Infer labels
labels = []
valid_filenames = []
for f in sorted(os.listdir(data_path)):
    if f.lower().endswith('.wav'):
        try:
            label = infer_instrument_from_filename(f)
            labels.append(map_class_to_id[label])
            valid_filenames.append(os.path.join(data_path, f))
        except ValueError as e:
            print(f"Skipping file {f}: {e}")
filenames = valid_filenames

# Print class distribution
class_counts = {instrument: labels.count(idx) for idx, instrument in enumerate(my_classes)}
print("Class distribution:", class_counts)

# Split data
if len(filenames) < 5:
    raise ValueError("Not enough files for train/val/test split. Need at least 5 files.")
train_files, test_files, train_labels, test_labels = train_test_split(
    filenames, labels, test_size=0.2, random_state=42, stratify=labels
)
train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.25, random_state=42, stratify=train_labels
)

# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))

# Map WAV loading
def load_wav_for_map(filename, label):
    wav = tf.py_function(load_wav_16k_mono, [filename], tf.float32)
    wav.set_shape([None])
    return wav, label

train_ds = train_ds.map(load_wav_for_map, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(load_wav_for_map, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(load_wav_for_map, num_parallel_calls=tf.data.AUTOTUNE)

# Extract YAMNet embeddings
def extract_embedding(wav_data, label):
    try:
        scores, embeddings, _ = yamnet_model(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        return embeddings, tf.repeat(label, num_embeddings)
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        return tf.zeros([1, 1024], dtype=tf.float32), tf.repeat(label, 1)

train_ds = train_ds.map(extract_embedding, num_parallel_calls=tf.data.AUTOTUNE).unbatch()
val_ds = val_ds.map(extract_embedding, num_parallel_calls=tf.data.AUTOTUNE).unbatch()
test_ds = test_ds.map(extract_embedding, num_parallel_calls=tf.data.AUTOTUNE).unbatch()

# Batch and prefetch
train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

# Define and compile model
my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,), dtype=tf.float32, name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(my_classes))
], name='instrument_classifier')

my_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

# Train model
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = my_model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=[callback]
)

# Evaluate model
loss, accuracy = my_model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Define ReduceMeanLayer
class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    def call(self, inputs):
        return tf.math.reduce_mean(inputs, axis=self.axis)

# Build serving model as a tf.Module
class ServingModel(tf.Module):
    def __init__(self, yamnet_model, classifier_model):
        super().__init__()
        self.yamnet_model = yamnet_model  # Track YAMNet model
        self.classifier_model = classifier_model  # Track classifier
        self.reduce_mean = ReduceMeanLayer(axis=0, name='classifier')

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32, name='audio')])
    def serving_fn(self, audio):
        # Ensure 1D input
        audio = tf.ensure_shape(audio, [None])
        # Extract embeddings
        _, embeddings, _ = self.yamnet_model(audio)
        # Apply classifier
        outputs = self.classifier_model(embeddings)
        # Reduce mean over frames
        outputs = self.reduce_mean(outputs)
        return {'classifier': outputs}

# Create and save serving model
serving_model = ServingModel(yamnet_model, my_model)
saved_model_path = './instrument_yamnet'
try:
    tf.saved_model.save(
        serving_model,
        saved_model_path,
        signatures={'serving_default': serving_model.serving_fn}
    )
    print(f"Model saved to {saved_model_path}")
except Exception as e:
    print(f"Error saving model: {e}")
    raise

# Load and test saved model
try:
    reloaded_model = tf.saved_model.load(saved_model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Test on a sample file
test_wav_file = os.path.join(data_path, 'Drum_1.wav')
if not os.path.exists(test_wav_file):
    print(f"Test file {test_wav_file} not found. Testing with first available file.")
    test_wav_file = filenames[0] if filenames else None
if test_wav_file:
    waveform = load_wav_16k_mono(tf.constant(test_wav_file))
    # Test with serving_default signature
    try:
        serving_results = reloaded_model.signatures['serving_default'](waveform)
        instrument = my_classes[tf.math.argmax(serving_results['classifier'])]
        print(f'[Your model] The main instrument in {os.path.basename(test_wav_file)} is: {instrument}')
    except Exception as e:
        print(f"Error during inference: {e}")

    # Compare with YAMNet raw predictions
    try:
        scores, embeddings, spectrogram = yamnet_model(waveform)
        class_scores = tf.reduce_mean(scores, axis=0)
        # Load YAMNet class names
        class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names = list(pd.read_csv(class_map_path)['display_name'])
        top_class = tf.math.argmax(class_scores)
        inferred_class = class_names[top_class]
        top_score = class_scores[top_class]
        print(f'[YAMNet] The main sound is: {inferred_class} ({top_score:.4f})')
        print(f'[Your model] The main instrument is: {instrument}')

        # Visualize waveform
        plt.figure(figsize=(10, 4))
        plt.plot(waveform.numpy())
        plt.title(f'Waveform: {os.path.basename(test_wav_file)}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.show()

        # Play audio
        display.Audio(waveform.numpy(), rate=16000)
    except Exception as e:
        print(f"Error during YAMNet comparison: {e}")
else:
    print("No test file available.")



## Gradio UI

import os
import numpy as np
import tensorflow as tf
import librosa
import gradio as gr
import socket
import warnings
from datetime import datetime
import soundfile as sf

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Install required libraries
try:
    import soundfile
except ImportError:
    !pip install -q tensorflow==2.11.* librosa soundfile gradio==4.44.0
    import soundfile

# Define classes 
my_classes = ['drums', 'guitar']

# Utility function to load and preprocess WAV file
def load_wav_16k_mono(audio):
    """Load a WAV file or Gradio audio input, resample to 16 kHz mono, and convert to a float tensor."""
    try:
        if isinstance(audio, str):
            # Handle file path (from fallback)
            # Validate WAV file
            try:
                info = sf.info(audio)
                print(f"Audio Info: Sample Rate={info.samplerate}, Channels={info.channels}, Duration={info.duration}s")
                if info.format != 'WAV':
                    return None, f"Error: File is not a valid WAV (format: {info.format})."
            except Exception as e:
                return None, f"Error validating WAV: {e}"
            # Load and resample to 16 kHz mono
            wav, sr = librosa.load(audio, sr=16000, mono=True)
        else:
            # Handle Gradio audio input (sample_rate, data)
            sr, wav = audio
            print(f"Gradio Audio Input: Sample Rate={sr}, Shape={wav.shape}, Dtype={wav.dtype}")
            # Convert to float32 if not already floating-point
            if not np.issubdtype(wav.dtype, np.floating):
                # Normalize int16 to [-1.0, 1.0]
                wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            if wav.ndim > 1:
                wav = librosa.to_mono(wav.T)
        
        # Check if waveform is valid
        if len(wav) == 0 or np.all(wav == 0):
            return None, "Error: Audio is empty or all zeros."
        
        # Ensure minimum length of 0.96 seconds (YAMNet requirement)
        min_samples = int(0.96 * 16000)
        if len(wav) < min_samples:
            wav = np.pad(wav, (0, max(0, min_samples - len(wav))), mode='constant')
        
        # Normalize to [-1.0, 1.0]
        max_val = np.max(np.abs(wav)) if np.max(np.abs(wav)) > 0 else 1.0
        wav = wav / max_val
        wav = tf.convert_to_tensor(wav, dtype=tf.float32)
        wav = tf.clip_by_value(wav, -1.0, 1.0)
        return wav, None
    except Exception as e:
        return None, f"Error loading audio: {e}"

# Function to predict instrument
def predict_instrument(audio, model_path='./instrument_yamnet'):
    """Predict the instrument for an uploaded audio file."""
    if audio is None:
        return "Error: No audio file uploaded. Please select a .wav file."

    # Load the saved model
    try:
        reloaded_model = tf.saved_model.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        return f"Error loading model: {e}\nEnsure './instrument_yamnet' exists."

    # Preprocess the audio
    waveform, error = load_wav_16k_mono(audio)
    if waveform is None:
        return error or "Error: Failed to process audio. Ensure the file is a valid .wav."
    if tf.reduce_all(waveform == 0):
        return "Error: Processed audio is all zeros. Ensure the file is a valid .wav."

    # Make prediction
    try:
        serving_results = reloaded_model.signatures['serving_default'](waveform)
        logits = serving_results['classifier']
        probabilities = tf.nn.softmax(logits).numpy()
        predicted_class_idx = tf.math.argmax(logits).numpy()
        predicted_instrument = my_classes[predicted_class_idx]

        # Format output
        output = f"Predicted Instrument: {predicted_instrument}\n\nConfidence Scores:\n"
        for class_name, prob in zip(my_classes, probabilities):
            output += f"- {class_name}: {prob:.4f}\n"
        return output
    except Exception as e:
        return f"Error during inference: {e}\nCheck model signature and input file."

# Fallback function to save predictions to a file
def predict_and_save(audio_path, model_path='./instrument_yamnet'):
    """Fallback: Predict and save results to a file if Gradio fails."""
    try:
        result = predict_instrument(audio_path, model_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"prediction_{timestamp}.txt"
        with open(output_file, 'w') as f:
            f.write(f"Audio File: {audio_path}\n\n{result}")
        print(f"Prediction saved to {output_file}")
        return result
    except Exception as e:
        error_msg = f"Error in fallback prediction: {e}"
        print(error_msg)
        return error_msg

# Function to check if a port is available
def is_port_available(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(('localhost', port)) != 0
    except Exception:
        return False

# Create and launch Gradio interface
def launch_gradio_app():
    try:
        # Find an available port
        port = 7840
        max_attempts = 5
        for i in range(max_attempts):
            if is_port_available(port):
                break
            print(f"Port {port} is in use, trying {port + 1}...")
            port += 1
            time.sleep(1)
        else:
            return ("Error: No available ports found (7860-7864). "
                    "Free up ports or check firewall settings.")

        # Define Gradio interface
        iface = gr.Interface(
            fn=predict_instrument,
            inputs=gr.Audio(type="numpy", label="Upload a .wav file"),
            outputs=gr.Markdown(label="Prediction Results"),
            title="Instrument Classifier",
            description="Upload a .wav file to predict drums or guitar.",
            allow_flagging="never"
        )

        # Launch Gradio locally
        print(f"Attempting to launch Gradio on http://127.0.0.1:{port}...")
        iface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=True,
            quiet=True
        )
        return f"Gradio app launched successfully on http://127.0.0.1:{port}"
    except Exception as e:
        return (f"Error launching Gradio: {e}\n"
                "Try: 1) Free ports 7860-7864, 2) Disable VPN/firewall, "
                "3) Restart runtime in Colab, 4) Use fallback prediction.")

# Run the app
if __name__ == "__main__":
    # Try Gradio launch
    result = launch_gradio_app()
    print(result)
    # Fallback to file-based prediction if Gradio fails
    if "Error" in result:
        print("\nFalling back to file-based prediction...")
        test_audio = "./new_audio/Guitar_7.wav"  # Update with your .wav file path
        if os.path.exists(test_audio):
            fallback_result = predict_and_save(test_audio)
            print(fallback_result)
        else:
            print(f"Error: Test audio {test_audio} not found. Please provide a valid .wav file.")
