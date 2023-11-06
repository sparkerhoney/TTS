from flask import Flask, request, send_file
from TTS.api import TTS

app = Flask(__name__)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=True)

@app.route('/synthesize', methods=['POST'])
def synthesize_text():
    data = request.json
    text = data['text']
    language = data.get('language', 'en')  # 기본값으로 'en'을 사용
    file_path = 'output.wav'

    # TTS 변환 수행
    tts.tts_to_file(text=text, file_path=file_path, language=language)

    # 생성된 오디오 파일을 반환
    return send_file(file_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
