        // Wait for the document to load before running the JavaScript
        document.addEventListener("DOMContentLoaded", function () {
            const industryTypeSelect = document.getElementById("industry-type");
            const subsectorsDiv = document.getElementById("subsectors");
            const subsectorSelect = document.getElementById("subsector");
        
            // Define subsectors for each major sector (customize as needed)
            const subsectors = {
                agriculture: ["Subsector A1", "Subsector A2", "Subsector A3"],
                service: ["Subsector S1", "Subsector S2", "Subsector S3"],
                manufacture: ["Subsector M1", "Subsector M2", "Subsector M3"],
                others: ["Subsector O1", "Subsector O2", "Subsector O3"],
            };
        
            // Function to update subsector options
            function updateSubsectors() {
                const selectedIndustryType = industryTypeSelect.value;
                if (selectedIndustryType === "india") {
                    subsectorsDiv.style.display = "none";
                } else {
                    subsectorsDiv.style.display = "block";
                    subsectorSelect.innerHTML = "";
        
                    // Populate the subsector options based on the selected industry type
                    subsectors[selectedIndustryType].forEach(subsector => {
                        const option = document.createElement("option");
                        option.value = subsector;
                        option.textContent = subsector;
                        subsectorSelect.appendChild(option);
                    });
                }
            }
        
            // Add an event listener to the industryTypeSelect
            industryTypeSelect.addEventListener("change", updateSubsectors);
        
            // Call the function initially to set the initial state
            updateSubsectors();
        });