from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
import pickle
import numpy as np
import pandas as pd
import xgboost
import joblib

from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
import librosa
model = xgboost.XGBClassifier()
model.load_model('model.json')
scaler_filename = "scaler.save"
sc = joblib.load(scaler_filename)
genre_map = {0:'blues',
             1:'classical',
            2:'country',
            3:'disco',
            4:'hiphop',
            5:'jazz',
            6:'metal',
            7:'pop',
            8:'reggae',
            9:'rock'}

def getdataf(filename):
    y, sr = librosa.load(filename)
    # fetching tempo

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    # fetching beats

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # chroma_stft

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    # rmse

    rmse = librosa.feature.rms(y=y)

    # fetching spectral centroid

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # spectral bandwidth

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # fetching spectral rolloff

    spec_rolloff = librosa.feature.spectral_rolloff(y=y + 0.01, sr=sr)[0]

    # zero crossing rate

    zero_crossing = librosa.feature.zero_crossing_rate(y)
    #     harmonics = [1./3, 1./2, 1, 2, 3, 4]
    #     S = np.abs(librosa.stft(y))
    #     fft_freqs = librosa.fft_frequencies(sr=sr)
    #     S_harm = librosa.interp_harmonics(S, freqs=fft_freqs, harmonics=harmonics, axis=0)
    #     S = np.abs(librosa.stft(y))
    #     fft_freqs = librosa.fft_frequencies(sr=sr)
    #     t_harmonics = librosa.interp_harmonics(S, freqs=fft_freqs, harmonics=harmonics, axis=0)

    #     C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A1')))
    #     freqs = librosa.cqt_frequencies(C.shape[0],
    #                                 fmin=librosa.note_to_hz('A1'))
    #     perceptual_CQT = librosa.perceptual_weighting(C**2,
    #                                               freqs,
    #                                               ref=np.max)
    # mfcc

    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # metadata dictionary

    metadata_dict = {'chroma_stft_mean': np.mean(chroma_stft), 'chroma_stft_var': np.var(chroma_stft),
                     'rms_mean': np.mean(rmse), 'rms_var': np.var(rmse),
                     'spectral_centroid_mean': np.mean(spec_centroid), 'spectral_centroid_var': np.var(spec_centroid),
                     'spectral_bandwidth_mean': np.mean(spec_bw), 'spectral_bandwidth_var': np.var(spec_bw),
                     'rolloff_mean': np.mean(spec_rolloff), 'rolloff_var': np.var(spec_rolloff),
                     'zero_crossing_rate_mean': np.mean(zero_crossing), 'zero_crossing_rate_var': np.var(zero_crossing),
                     # 'harmony_mean':np.mean(t_harmonics),'harmony_var':np.var(t_harmonics),
                     # 'perceptr_mean':np.mean(perceptual_CQT),'perceptr_var':np.var(perceptual_CQT),
                     'tempo': tempo}

    for i in range(1, 21):
        metadata_dict.update({'mfcc' + str(i) + '_mean': np.mean(mfcc[i - 1])})
        metadata_dict.update({'mfcc' + str(i) + '_var': np.var(mfcc[i - 1])})
    df_meta = pd.DataFrame(metadata_dict, index=[0])
    return df_meta


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])

def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        m_df=getdataf(file)
        m_df = sc.transform(m_df)

        output=model.predict(m_df)
        #file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        return genre_map[output[0]]
    return render_template('index.html', form=form)




if __name__ == '__main__':
    app.run(debug=True)