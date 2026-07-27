# Vietnamese TextVoice Studio

<div align="center">
  <img src="https://img.shields.io/badge/Python-TTS-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-Web%20Studio-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/Language-Vietnamese-DA251D?style=for-the-badge" alt="Vietnamese"/>
</div>

Web studio tạo giọng nói tiếng Việt và clone giọng bằng **Gwen-TTS**. Ứng dụng Flask chạy một engine TTS duy nhất, có sẵn 10 preset voice, hỗ trợ audio tham chiếu tự tải lên, điều chỉnh tham số sinh, quy tắc phát âm và hàng đợi cho request dài.

> Dùng audio khi có quyền sử dụng giọng nói và nội dung trong audio.

## Tính năng

- Clone giọng tiếng Việt bằng `g-group-ai-lab/gwen-tts-0.6B`.
- 10 preset voice có thể nghe thử và dùng trực tiếp.
- Clone từ audio riêng với transcript tham chiếu bắt buộc.
- Chuẩn hóa văn bản trước khi synthesize: xuống dòng, danh sách, dấu câu, khoảng trắng; số, ngày tháng và viết tắt được chuẩn hóa thêm khi có `sea-g2p`.
- Tùy chỉnh tốc độ, sampling, `temperature`, `top_p`, `top_k`, `repetition_penalty` và `max_new_tokens`.
- Thêm, nhập hoặc xuất quy tắc phát âm `.json`/`.txt`.
- Lưu lịch sử synthesize và cài đặt phát âm cục bộ.
- Tùy chọn ASR để nhận transcript audio tham chiếu hoặc chép lời audio dài.

## Yêu cầu

- Python 3.
- GPU CUDA khả dụng cho Gwen-TTS. App sẽ hiển thị engine chưa sẵn sàng nếu không tìm thấy CUDA.
- Môi trường Python có PyTorch tương thích CUDA và `torch.cuda.is_available()` trả về `True`.
- Internet trong lần chạy đầu để tải model từ Hugging Face.

Gwen-TTS cần audio tham chiếu dài từ 2 đến 18 giây. Khoảng 3 đến 10 giây, một người nói và ít nhiễu cho kết quả ổn định nhất.

## Cài đặt và chạy

Ví dụ cho Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -U qwen-tts sea-g2p
python webapp/app.py
```

Mở `http://127.0.0.1:8386`.

`requirements.txt` chỉ chứa dependency web và xử lý audio cơ bản. `qwen-tts` cùng stack PyTorch/CUDA được cài riêng vì phụ thuộc GPU theo từng máy.

Kiểm tra CUDA trước khi chạy:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Kết quả cuối phải là `True`.

## Dùng Studio

1. Mở `/studio/gwen`.
2. Chọn một preset voice hoặc mở `/voices` để tải audio mẫu và tạo clone từ audio riêng.
3. Nếu dùng audio riêng, tải file `.wav`, `.mp3`, `.m4a`, `.flac` hoặc `.ogg`, rồi nhập đúng transcript của câu trong audio.
4. Nhập nội dung cần đọc, tối đa 5.000 ký tự.
5. Bấm **Chuyển Đổi**. Audio đầu ra là WAV, có thể phát, tải xuống hoặc sao chép URL.

Preset voice bỏ qua bước upload và transcript vì audio cùng transcript đã có sẵn trong project.

### Tinh chỉnh phát âm

Mục **Phát âm** thay chuỗi trước khi gửi text vào model. Dùng khi model đọc sai tên riêng, chữ viết tắt hoặc thuật ngữ.

File import hỗ trợ hai dạng:

```json
[
  { "from": "KPI", "to": "cây pi ai" },
  { "from": "AI", "to": "ây ai" }
]
```

Hoặc file `.txt`, mỗi dòng một cặp:

```text
KPI -> cây pi ai
AI|ây ai
TP.HCM = thành phố Hồ Chí Minh
```

## Cấu hình môi trường

### Web server

- `TTS_WEBAPP_HOST`: host bind; mặc định `0.0.0.0`.
- `TTS_WEBAPP_PORT`: port; mặc định `8386`.
- `TTS_WEBAPP_DEBUG`: đặt `1` để bật Flask debug; mặc định tắt.
- `TTS_WEBAPP_RELOADER`: đặt `1` để bật reloader; mặc định tắt.

Ví dụ:

```bash
TTS_WEBAPP_HOST=127.0.0.1 TTS_WEBAPP_PORT=9000 python webapp/app.py
```

### Gwen-TTS

- `GWEN_MODEL_ID`: model mặc định; mặc định `g-group-ai-lab/gwen-tts-0.6B`.
- `GWEN_MODEL_CHOICES`: thêm model vào UI, phân cách bằng `;`. Có thể dùng `Nhãn=model-id`.
- `GWEN_DTYPE`: `bfloat16`, `float16` hoặc `float32`; mặc định `bfloat16`.
- `GWEN_ATTN_IMPLEMENTATION`: mặc định `flash_attention_2`. Khi model không load được với lựa chọn này, app thử lại bằng `sdpa`.

Ví dụ:

```bash
export GWEN_DTYPE=float16
export GWEN_ATTN_IMPLEMENTATION=sdpa
python webapp/app.py
```

### ASR tùy chọn

ASR tắt mặc định để tránh xung đột dependency/model với Gwen-TTS. Bật ASR để nhận transcript tham chiếu bằng Whisper hoặc dùng trang `/asr` với VibeVoice-ASR.

```bash
python -m pip install -U transformers accelerate sentencepiece
TTS_ENABLE_ASR=1 python webapp/app.py
```

Biến liên quan:

- `TTS_ENABLE_ASR`: đặt `1` để bật ASR.
- `ASR_MODEL_ID`: Whisper cho transcript tham chiếu; mặc định `openai/whisper-small`.
- `ASR_LANGUAGE`: ngôn ngữ Whisper; mặc định `vi`.
- `ASR_CHUNK_LENGTH_S`: chunk Whisper, từ 8 đến 30 giây; mặc định `18`.
- `VIBEVOICE_ASR_MODEL_ID`: model cho `/asr`; mặc định `microsoft/VibeVoice-ASR-HF`.

## API

### Trạng thái engine

```http
GET /api/tts/status
```

Trả trạng thái Gwen-TTS, GPU/CPU và trạng thái ASR.

### Sinh audio

```http
POST /api/tts/generate
Content-Type: multipart/form-data
```

Trường cần có:

- `text`: nội dung cần đọc.
- Một trong hai cách chọn giọng:
  - `preset_voice_id`, ví dụ `yen_nhi`; hoặc
  - `reference_audio` và `reference_text` cho clone audio riêng.

Trường tùy chọn gồm `model_key`, `custom_model`, `speed`, các trường `gwen_*`, cặp `pronunciation_from`/`pronunciation_to`, và `async`.

Ví dụ clone audio riêng:

```bash
curl -X POST http://127.0.0.1:8386/api/tts/generate \
  -F 'text=Xin chào, đây là bản ghi thử.' \
  -F 'reference_audio=@sample.wav' \
  -F 'reference_text=Xin chào, đây là câu trong audio mẫu.'
```

Thêm `-F 'async=1'` để nhận `202 Accepted` cùng `job_id`. Theo dõi job qua:

```http
GET /api/tts/jobs/<job_id>
```

### Voice, cài đặt và lịch sử

- `GET /outputs/<filename>`: phát file WAV đã sinh.
- `GET /api/history`: lấy tối đa 50 lần synthesize gần nhất.
- `GET /api/settings/load`: lấy quy tắc phát âm và text gần nhất.
- `POST /api/settings/save`: lưu quy tắc phát âm và text gần nhất.
- `POST /api/pronunciation/upload`: đọc file quy tắc `.json` hoặc `.txt`.
- `POST /api/pronunciation/export`: xuất quy tắc thành JSON.

### ASR

Các endpoint sau chỉ hoạt động khi `TTS_ENABLE_ASR=1`:

- `POST /api/tts/transcribe-reference`: nhận transcript từ `reference_audio`.
- `POST /api/asr/transcribe`: chép lời từ trường `audio`, hỗ trợ `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`.

Giới hạn upload hiện tại là 32 MB.

## Dữ liệu runtime

App tự tạo các thư mục và file sau khi cần:

- `webapp/runtime/references/`: audio tham chiếu đã upload.
- `webapp/runtime/generated/`: WAV đã sinh.
- `webapp/data/user_settings.json`: quy tắc phát âm và text gần nhất.
- `webapp/data/tts_history.json`: lịch sử synthesize, tối đa 50 mục.
- `models/`: thư mục model cục bộ dành cho runtime.

Các file runtime có thể chứa audio và nội dung người dùng. Không đưa chúng lên repository hoặc chia sẻ khi chưa có quyền.

## Cấu trúc

```text
webapp/
├── app.py                 # Flask routes, API và hàng đợi synthesize
├── tts_service.py         # Gwen-TTS, chuẩn hóa text/audio và ASR
├── data/                  # preset voice, cài đặt và lịch sử cục bộ
├── static/                # CSS, logo và audio preset
└── templates/             # Studio, danh sách voice và ASR

tests/                     # unittest cho API và helper TTS
requirements.txt           # dependency web/audio tối thiểu
```

## Kiểm tra

```bash
python -m unittest discover -s tests -v
```

Test không tải model Gwen-TTS hoặc chạy inference GPU thật.
