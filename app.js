const express = require("express");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 3000;

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// Set EJS
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

// Load oceanData.json
let oceanData = {};
try {
  const rawData = fs.readFileSync(path.join(__dirname, "data", "oceanData.json"));
  oceanData = JSON.parse(rawData);
} catch (err) {
  console.error("Error loading oceanData.json:", err);
}

// Default route
app.get("/", (req, res) => {
  // Use `req.query.region` which can be a string or an array
  const { parameter = "" } = req.query;
  let selectedRegion = req.query.region;
  
  // Normalize `selectedRegion` to always be an array for consistent logic
  if (typeof selectedRegion === 'string') {
    selectedRegion = [selectedRegion];
  } else if (!Array.isArray(selectedRegion)) {
    selectedRegion = [];
  }

  const datasets = [];
  const labels = Object.values(oceanData)[0]?.years || [];
  
  // Build the array of datasets for Chart.js
  selectedRegion.forEach(region => {
    if (oceanData[region] && oceanData[region][parameter]) {
      datasets.push({
        label: `${region} - ${parameter}`,
        data: oceanData[region][parameter],
        borderColor: `rgb(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255})`, // Random color for each line
        backgroundColor: `rgba(0, 0, 0, 0.1)`,
        fill: false,
        tension: 0.3,
      });
    }
  });

  const chartData = {
    labels,
    datasets
  };

  res.render("layout", {
    regions: Object.keys(oceanData), // sidebar dropdown
    chartData,
    selectedRegion, // Pass the array to the view
    parameter
  });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
