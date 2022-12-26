var buttonRecord = document.getElementById("record");
var buttonStop = document.getElementById("stop");

buttonStop.disabled = true;

//progress bar functionality
function progress(timeleft, timetotal, $element) {
  var progressBarWidth = (timeleft * $element.width()) / timetotal;
  $element
    .find("div")
    .animate({ width: progressBarWidth }, 500)
    .html(timeleft + " seconds to go");
  if (timeleft > 0) {
    setTimeout(function () {
      progress(timeleft - 1, timetotal, $element);
    }, 1000);
  }
  if (timeleft == 0) {
    stopRecording();
  }
  console.log("Time Left Data : ", timeleft);
}
buttonRecord.onclick = function () {
  progress(30, 30, $("#progressBar"));
  // var url = window.location.href + "record_status";
  buttonRecord.disabled = true;
  buttonStop.disabled = false;

  // disable download link
  var downloadLink = document.getElementById("download");
  downloadLink.text = "";
  downloadLink.href = "";

  // XMLHttpRequest
  var xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function () {
    if (xhr.readyState == 4 && xhr.status == 200) {
      // alert(xhr.responseText);
    }
  };
  xhr.open("POST", "/record_status");
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xhr.send(JSON.stringify({ status: "true" }));
};

function stopRecording() {
  buttonRecord.disabled = false;
  buttonStop.disabled = true;

  // XMLHttpRequest
  var xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function () {
    if (xhr.readyState == 4 && xhr.status == 200) {
      // alert(xhr.responseText);

      // enable download link
      var downloadLink = document.getElementById("download");
      downloadLink.text = "Download Video";
      downloadLink.href = "/static/video.avi";
    }
  };
  xhr.open("POST", "/record_status");
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xhr.send(JSON.stringify({ status: "false" }));
}
// buttonStop.onclick = function () {
//   buttonRecord.disabled = false;
//   buttonStop.disabled = true;

//   // XMLHttpRequest
//   var xhr = new XMLHttpRequest();
//   xhr.onreadystatechange = function () {
//     if (xhr.readyState == 4 && xhr.status == 200) {
//       // alert(xhr.responseText);

//       // enable download link
//       var downloadLink = document.getElementById("download");
//       downloadLink.text = "Download Video";
//       downloadLink.href = "/static/video.avi";
//     }
//   };
//   xhr.open("POST", "/record_status");
//   xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
//   xhr.send(JSON.stringify({ status: "false" }));
// };
