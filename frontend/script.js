async function analyzeSentiment() {
    const text = document.getElementById("textInput").value.trim();

    if (text === "") {
        alert("Please enter some text!");
        return;
    }

    try {
        const response = await fetch(
            `http://127.0.0.1:8000/predict?text=${encodeURIComponent(text)}`,
            {
                method: "POST"
            }
        );

        const data = await response.json();

        document.getElementById("sentiment").innerText = data.sentiment;
        document.getElementById("confidence").innerText =
            (data.confidence * 100).toFixed(2) + "%";

        document.getElementById("resultBox").classList.remove("hidden");

    } catch (error) {
        alert("Backend not running!");
        console.error(error);
    }
}
