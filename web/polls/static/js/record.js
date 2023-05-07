const $audioEl = document.getElementById("audio");
const $btn = document.getElementById("record");
const $stopBtn = document.getElementById("stop");
const $uploadBtn = document.getElementById("upload");
let isRecording = false;
const audioArray = [];
let mediaRecorder = null;

$btn.onclick = async function (event) {
  if (!isRecording) {
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });

    mediaRecorder = new MediaRecorder(mediaStream);
    mediaRecorder.start(1000);

    mediaRecorder.ondataavailable = addToAudioArray;
    mediaRecorder.onstop = stopRecording;

    isRecording = true;
  } else {
    stopRecording();
  }
};

$stopBtn.onclick = function (event) {
  if (isRecording) {
    stopRecording();
  }
};

$uploadBtn.onclick = function (event) {
  uploadAudioFile();
};

async function uploadAudioFile() {
  if (audioArray.length > 0) {
    const blob = new Blob(audioArray, { type: "audio/webm; codecs=opus" });
    const formData = new FormData();
    formData.append("audio_file", blob, "audio.webm");

    const csrfToken = document.cookie.match(/csrftoken=([\w-]+)/)[1];
    const response = await fetch("/mypage/", {
      method: "POST",
      body: formData,
      headers: {
        "X-CSRFToken": csrfToken,
      },
    });

    if (response.ok) {
      console.log("File uploaded successfully!");
    } else {
      console.error("File upload failed with status", response.status);
    }
  } else {
    console.log("No audio data recorded.");
  }

  // 초기화
  audioArray.splice(0);
  isRecording = false;
  $audioEl.pause();
  $audioEl.removeAttribute("src");
  $audioEl.load();
  $uploadBtn.disabled = true;
}

function addToAudioArray(event) {
  if (event.data) {
    // event.data가 undefined가 아닐 경우에만 추가
    audioArray.push(event.data);
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    mediaRecorder = null;
  }
  console.log("audioArray:", audioArray);
  $uploadBtn.disabled = false;
  // $audioEl에 재생할 오디오 데이터를 할당합니다.
  const blob = new Blob(audioArray, { type: "audio/webm; codecs=opus" });
  $audioEl.src = window.URL.createObjectURL(blob);
}
