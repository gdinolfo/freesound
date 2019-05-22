from flask import Flask, render_template, request,redirect
from IPython.display import Audio
import librosa
import io
import soundfile as sf
import pickle
import numpy as np
from collections import Counter
import pandas as pd

def get_subarrays(row):
    return [row[pos:pos + 3,] for pos in range(0, len(row),3)]

labels = {'Saxophone': 0,
 'Glockenspiel': 1,
 'Cello': 2,
 'Knock': 3,
 'Gunshot_or_gunfire': 4,
 'Hi-hat': 5,
 'Laughter': 6,
 'Flute': 7,
 'Telephone': 8,
 'Bark': 9,
 'Scissors': 10,
 'Gong': 11,
 'Microwave_oven': 12,
 'Shatter': 13,
 'Harmonica': 14,
 'Bass_drum': 15,
 'Oboe': 16,
 'Bus': 17,
 'Tambourine': 18,
 'Keys_jangling': 19,
 'Electric_piano': 20,
 'Clarinet': 21,
 'Fireworks': 22,
 'Meow': 23,
 'Double_bass': 24,
 'Cough': 25,
 'Acoustic_guitar': 26,
 'Violin_or_fiddle': 27,
 'Snare_drum': 28,
 'Squeak': 29,
 'Finger_snapping': 30,
 'Writing': 31,
 'Trumpet': 32,
 'Drawer_open_or_close': 33,
 'Cowbell': 34,
 'Tearing': 35,
 'Fart': 36,
 'Chime': 37,
 'Burping_or_eructation': 38,
 'Computer_keyboard': 39,
 'Applause': 40}
inverted_mapping = {v:k for k,v in labels.items()}

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10),min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


app = Flask(__name__)

model = pickle.load(open('aihua_model.pkl','rb'))
model._make_predict_function()

@app.route("/")
def index():
    """Return the homepage."""
    return render_template("index.html")


@app.route('/upload', methods = ['POST'])
def api_message():
    myfile = request.files.get('myWav').read() # just bytes
    data, samplerate = sf.read(io.BytesIO(myfile))
    mask = envelope(data,samplerate,0.0005)
    new_sig = data[mask]
    S = librosa.feature.melspectrogram(y=new_sig, sr=samplerate)
    np.save("new.npy", S)
    data = np.load("new.npy").T
    data_slice = get_subarrays(data)
    xs = []
    for array_set in data_slice:
        if array_set.shape == (3,128):  # The network take input of shape (3, 128, 1)
            xs.append(array_set)
    features = np.asarray(xs)
    xs_reshape = features.reshape(features.shape[0], 3, 128, 1)
    pred_labels = [inverted_mapping.get(x) for x in model.predict_classes(xs_reshape)] # going to return a prediction for each of the 33
    label = Counter(pred_labels).most_common(1)[0][0]
    # return label
    # need to get the audio file procecessed
    # then load the model
    # then passt eh proc'd file into the model
    # return redirect template to some template with the var for the prediccted class
    # also save the file, so you can show a audio player in the HTML
    print(label)
    return render_template("index.html", post=label)



if __name__ == '__main__':
	app.run(debug=True)
