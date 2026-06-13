console.log("Script loaded");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");

imageInput.addEventListener("change", () => {

    const file = imageInput.files[0];

    preview.src = URL.createObjectURL(file);

});

async function predict() {

    try {

        const file = imageInput.files[0];

        if (!file) {
            alert("Please select an image");
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        document.getElementById("result").innerHTML = `
            <div class="result-card">
                <h2>Classification Result</h2>
                <h3>${data.class}</h3>
                <p>Confidence: ${data.confidence}%</p>
            </div>
        `;

    } catch (error) {

        console.log(error);

        document.getElementById("result").innerHTML =
            "<h3>Prediction Failed</h3>";
    }
}