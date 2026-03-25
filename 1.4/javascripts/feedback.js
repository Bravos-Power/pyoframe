document$.subscribe(function () {
  const feedback = document.forms.feedback;
  if (feedback === undefined) return;
  
  feedback.hidden = false;

  feedback.addEventListener("submit", function (ev) {
    ev.preventDefault();

    feedback.firstElementChild.disabled = true;

    const data = ev.submitter.getAttribute("data-md-value");

    if (data == 0) {
      const commentElement = document.getElementById("comments");
      commentElement.style.display = "block";
    }

    const note = feedback.querySelector(
      ".md-feedback__note [data-md-value='" + data + "']"
    );
    if (note) {
      note.hidden = false;
    }
  });
});
