from os import sep
from pyngrok import ngrok

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
from flask import Flask, render_template, url_for, request
# from werkzeug import secure_filename
from tensorflow.keras import initializers
from keras.models import Sequential
from keras.layers import Dense
from keras_adabound import AdaBound
from numpy.random import seed

app = Flask(__name__)

# tunnels = ngrok.get_tunnels()

@app.route("/")
def index():
    return render_template('dashboard.html')

@app.route("/indikator/")
def indikator():
    return render_template('indikator.html')
    
@app.route("/prediksi/")
def prediksi():
    return render_template('form_2.html')
    
@app.route("/tes/", methods=['GET', 'POST'])
def tes():
    if request.method == 'POST':
        f = request.files['data'] 
        print(f)
        # f.save(secure_filename(f.filename))
        return 'file uploaded successfully'
    
    # rd=pd.read_csv('data/15 indikator baru 2020.csv')

    # dt=list(csv.reader('data/15 indikator baru 2020.csv', sep))

    # print(dt)

    # # for line in dt:
    # #     print(line)
    return render_template('result.html')

@app.route("/proses/", methods=['GET', 'POST'])
def proses():
    tahun = request.form['tahun']
    bulan = request.form['bulan']
    Impor = float(request.form['Impor'])
    Ekspor = float(request.form['Ekspor'])
    Cadangan_Devisa = float(request.form['Cadangan_Devisa'])
    IHSG = float(request.form['IHSG'])
    Rasio_Suku_Bunga_Simpanan_dan_Pinjaman = float(request.form['Rasio_Suku_Bunga_Simpanan_dan_Pinjaman'])
    Suku_Bunga_Simpanan_Riil = float(request.form['Suku_Bunga_Simpanan_Riil'])
    Selisih_BI_rate_dan_FED_rate = float(request.form['Selisih_BI_rate_dan_FED_rate'])
    Simpanan_Bank = float(request.form['Simpanan_Bank'])
    Nilai_Tukar_Riil = float(request.form['Nilai_Tukar_Riil'])
    Nilai_Tukar_Perdagangan = float(request.form['Nilai_Tukar_Perdagangan'])
    M1 = float(request.form['M1'])
    Rasio_M2_terhadap_Cadangan_Devisa = float(request.form['Rasio_M2_terhadap_Cadangan_Devisa'])
    M2_Multiplier = float(request.form['M2_Multiplier'])

    dt = []

    dt = {
            'tahun': tahun,
            'bulan': bulan,
            'Impor': Impor,
            'Ekspor': Ekspor,
            'Cadangan_Devisa': Cadangan_Devisa,
            'IHSG': IHSG,
            'Rasio_Suku_Bunga_Simpanan_dan_Pinjaman': Rasio_Suku_Bunga_Simpanan_dan_Pinjaman,
            'Suku_Bunga_Simpanan_Riil': Suku_Bunga_Simpanan_Riil,
            'Selisih_BI_rate_dan_FED_rate': Selisih_BI_rate_dan_FED_rate,
            'Simpanan_Bank': Simpanan_Bank,
            'Nilai_Tukar_Riil': Nilai_Tukar_Riil,
            'Nilai_Tukar_Perdagangan': Nilai_Tukar_Perdagangan,
            'M1': M1,
            'Rasio_M2_terhadap_Cadangan_Devisa': Rasio_M2_terhadap_Cadangan_Devisa,
            'M2_Multiplier': M2_Multiplier
        }
    
    data = {'Impor': (Impor-7341.924)/5002.848,
            'Ekspor': (Ekspor-8304.717)/4800.132,
            'Cadev': (Cadangan_Devisa-53760.44)/40992.62,
            'IHSG': (IHSG-2096.18)/2013.639,
            'SBSimPin': (Rasio_Suku_Bunga_Simpanan_dan_Pinjaman-1.493616)/0.332593,
            'SBSimRiil': (Suku_Bunga_Simpanan_Riil-12.09453)/7.636004,
            'BI&FED': (Selisih_BI_rate_dan_FED_rate-8.545691)/7.868711,
            'SimBank': (Simpanan_Bank-1396419)/1411532,
            'NTRiil': (Nilai_Tukar_Riil-7324.375)/5578.899,
            'NTdag': (Nilai_Tukar_Perdagangan-1.221927)/0.211243,
            'M1': (M1-424080.9)/426535.6,
            'M2CD': (Rasio_M2_terhadap_Cadangan_Devisa-0.003746)/0.001197,
            'M2M': (M2_Multiplier-10.55653)/1.305534}     
    features = pd.DataFrame(data, index=[0])

    df = features

    X=pd.read_csv('data/variabel smote.csv', sep=';')
    Y=pd.read_csv('data/Target smote.csv', sep=';')

    initializer1 = tf.keras.initializers.GlorotUniform(seed=1)
    initializer2 = tf.keras.initializers.GlorotUniform(seed=2)
    
    model = Sequential()
    model.add(Dense(7, input_dim=13, activation='sigmoid', kernel_initializer=initializer1))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer2))

    optimizer = AdaBound(lr=0.1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(X, Y, epochs=100, batch_size=128)

    prediction = np.round_(model.predict(df))
    dt['pred'] = prediction

    if prediction == 0:
        dt['ket'] = 'Tidak terdeteksi ada sinyal krisi keuangan'
    else:
        dt['ket'] = 'Terdeteksi krisis keuangan'
    
    return render_template('result.html', dt=dt)

if __name__ == "__main__":
    app.run(debug=True)