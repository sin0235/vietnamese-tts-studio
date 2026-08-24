# Gwen Text Input Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bảo toàn nghĩa và ranh giới câu của text tiếng Việt trước khi gửi Gwen-TTS, đồng thời giữ mọi chunk không vượt 320 ký tự.

**Architecture:** Giữ pipeline hiện tại và sửa tại ba điểm chịu trách nhiệm chung: spacing dấu câu, fallback số và chunker. Structured token được che tạm chỉ trong lúc chỉnh spacing; parser số từ chối dữ liệu vượt phạm vi thay vì đọc sai; inference ưu tiên ngôn ngữ Vietnamese theo API Gwen chính thức.

**Tech Stack:** Python standard library, `unittest`, Flask service hiện có.

**Spec:** Báo cáo rà soát trong phiên ngày 2026-08-24.

## Global Constraints

- Không thêm dependency, schema, route hoặc public response field.
- Giữ diff trong `webapp/tts_service.py` và test liên quan.
- Quy tắc phát âm vẫn chạy trước SEA-G2P/fallback.
- Mỗi thay đổi behavior phải có regression test đã quan sát fail trước khi sửa source.

---

### Task 1: Bảo toàn structured token qua punctuation spacing

**Files:**
- Modify: `webapp/tts_service.py:471-495,746-809`
- Test: `tests/test_api_errors.py`

**Interfaces:**
- Consumes: `text: str` của `_normalize_tts_prompt_text()` và `_normalize_gwen_text()`.
- Produces: `_normalize_punctuation_spacing(text: str) -> str`, giữ nguyên decimal, email, URL, version/IP token.

- [x] **Step 1: Viết test fail cho `12,5%`, email và URL qua full preprocessing.**
- [x] **Step 2: Chạy riêng test và xác nhận output hiện tại chèn khoảng trắng vào structured token.**
- [x] **Step 3: Thêm helper chỉ chỉnh spacing ngoài structured token; dùng helper ở cả hai tầng.**
- [x] **Step 4: Chạy lại test và xác nhận pass.**

### Task 2: Sửa fallback số và giới hạn miền hỗ trợ

**Files:**
- Modify: `webapp/tts_service.py:234-377,499-546`
- Test: `tests/test_tts_service_helpers.py`

**Interfaces:**
- Consumes: numeric token từ `_GWEN_NUMBER_TOKEN_PATTERN`.
- Produces: text đọc tiếng Việt nhưng giữ dấu câu; số bắt đầu bằng `0` đọc từng chữ số; token vượt 18 chữ số hoặc dotted chain không hợp lệ được giữ nguyên.

- [x] **Step 1: Viết test fail cho dấu câu sau số, decimal comma/mixed separator, số điện thoại, IP/version và số 5.000 chữ số.**
- [x] **Step 2: Chạy riêng test và xác nhận fail đúng các output đã tái hiện.**
- [x] **Step 3: Buộc numeric match kết thúc bằng chữ số, phân tích separator hữu hạn và từ chối số vượt phạm vi trước `int()`.**
- [x] **Step 4: Chạy lại test và xác nhận pass.**

### Task 3: Bảo đảm hard limit của chunk

**Files:**
- Modify: `webapp/tts_service.py:691-733`
- Test: `tests/test_tts_service_helpers.py`

**Interfaces:**
- Consumes: normalized text và `max_chars`.
- Produces: `list[str]` trong đó mọi phần tử có `len(chunk) <= max_chars`, kể cả token không có khoảng trắng.

- [x] **Step 1: Viết test fail với 700 ký tự liền và 400 emoji.**
- [x] **Step 2: Chạy riêng test và xác nhận chunk hiện tại vượt giới hạn.**
- [x] **Step 3: Cắt riêng oversized word bằng slice `max_chars`, không đổi logic câu/từ bình thường.**
- [x] **Step 4: Chạy lại test và xác nhận pass.**

### Task 4: Ưu tiên language Vietnamese khi gọi Gwen

**Files:**
- Modify: `webapp/tts_service.py:1689-1765,1800-1825`
- Modify: `README.md:253-254`
- Modify: `README.en.md:253-254`
- Test: `tests/test_tts_service_helpers.py:316-478`

**Interfaces:**
- Consumes: API `generate_voice_clone(...)` hiện có.
- Produces: call đầu tiên có `language="Vietnamese"`; runtime không nhận tham số vẫn fallback sang direct call không language.

- [x] **Step 1: Đổi expectation test để call đầu tiên phải có `language="Vietnamese"` và test fallback phải bỏ language ở lần hai.**
- [x] **Step 2: Chạy test, xác nhận fail vì call plan hiện tại ngược thứ tự.**
- [x] **Step 3: Đảo call plan và cập nhật runtime note đúng với mode thực tế.**
- [x] **Step 4: Đồng bộ thứ tự fallback trong README tiếng Việt và tiếng Anh.**
- [x] **Step 5: Chạy lại test và xác nhận pass.**

### Task 5: Xác minh tổng thể

**Files:**
- Modify: không có nếu kiểm tra pass.

**Interfaces:**
- Consumes: toàn bộ diff.
- Produces: bằng chứng compile, test, diff sạch whitespace và không chạm file ngoài phạm vi.

- [x] **Step 1: Chạy `python -m unittest discover -s tests -v` trong environment tạm có `requirements.txt`.**
- [x] **Step 2: Chạy `python -m py_compile` với `PYTHONPYCACHEPREFIX` ngoài repository.**
- [x] **Step 3: Chạy `git diff --check`, đọc `git diff` và kiểm tra `git status --short`.**
