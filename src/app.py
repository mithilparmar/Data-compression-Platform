from flask import Flask,render_template, request, send_file
from Compressor import Compressor
from DeCompressor import DeCompressor
app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/compression",methods=('GET', 'POST'))
def compression_pages():
    
    return render_template("pages/compression.html")

@app.route("/compress_video",methods=('GET', 'POST'))
def compress_video():
    
    print(request.form)
    request.files["file"].save(dst="original_video.mp4")

    c = Compressor(0.5)
    c.compress_vid("original_video.mp4")
    return send_file('../demo.mp4')


@app.route("/decompression",methods=('GET', 'POST'))
def decompression_pages():
    
    return render_template("pages/decompression.html")

@app.route("/decompress_video",methods=('GET', 'POST'))
def decompress_video():
    
    print(request.form)
    request.files["file"].save(dst="compressed.mp4")

    c = DeCompressor("compressed.mp4","return_video.mp4")
    c.decompress()
    return send_file('../return_video.mp4')

