document$.subscribe(function () {
  const feedback = document.forms.feedback;
  if (feedback === undefined) return;
  
  feedback.hidden = false;

  function recordFeedback(value) {
    if (
      !window.goatcounter ||
      typeof window.goatcounter.count !== "function"
    ) {
      console.warn("goatcounter not found, feedback not recorded");
      return;
    }

    // Must use == not ===
    const vote = value == 0 ? "sad" : "happy";

    window.goatcounter.count({
      path: function(p) { return 'feedback-' + vote + '-' + p },
      title: "Feedback " + vote,
      event: true,
    });
  }

  feedback.addEventListener("submit", function (ev) {
    ev.preventDefault();

    feedback.firstElementChild.disabled = true;

    const data = ev.submitter.getAttribute("data-md-value");
    recordFeedback(data);

    // Must use == not ===
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
