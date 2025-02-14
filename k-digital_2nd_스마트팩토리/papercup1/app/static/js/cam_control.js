// 페이지가 나갈 때 웹캠 종료
window.addEventListener("beforeunload", function () {
  var video = document.getElementById("video_feed");
  video.src = ""; // 스트리밍 멈춤
});
