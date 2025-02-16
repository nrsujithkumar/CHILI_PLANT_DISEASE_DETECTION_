document.addEventListener('DOMContentLoaded', function () {
    const plusButton = document.getElementById('plus-button');
    const tooltip = document.getElementById('tooltip');
    const fileInput = document.getElementById('file-input');
    const processingMessage = document.getElementById('processing-message');
  
    // Show tooltip on mouse enter and hide on leave
    plusButton.addEventListener('mouseenter', () => {
      tooltip.style.display = 'block';
    });
    plusButton.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
    });
  
    // Trigger file input when plus button is clicked
    plusButton.addEventListener('click', () => {
      fileInput.click();
    });
  
    // When a file is selected, display processing message and send file to backend
    fileInput.addEventListener('change', () => {
      const file = fileInput.files[0];
      if (file) {
        processingMessage.style.display = 'block';
  
        const formData = new FormData();
        formData.append('file', file);
  
        // Send the image to the backend '/predict' endpoint
        fetch('/predict', {
          method: 'POST',
          body: formData
        })
        .then(response => response.json())
        .then(data => {
          processingMessage.style.display = 'none';
          // Open a new tab with the result page; pass the prediction as a query parameter
          const resultUrl = `/result?prediction=${encodeURIComponent(data.prediction)}`;
          window.open(resultUrl, '_blank');
        })
        .catch(error => {
          console.error('Error:', error);
          processingMessage.innerText = 'Error processing the image.';
        });
      }
    });
  });
  