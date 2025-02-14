window.onload = function () {
  var video = document.getElementById("video_feed");
  video.src = "{{ url_for('video_feed') }}"; // 스트리밍 URL 새로 설정
};
