# IITK Hackathon Buzzstop

## Title : Air Quality Index

## Team Name: AI AVENGERS

## Problem Statement :  Air Quality monitoring and Prediction
 
---
###  Challenge : 
 
In addition, air pollution negatively affects the urban ecosystem, damaging vegetation, reducing biodiversity, and contributing to climate change. Poor air quality has significant impacts on public health, contributing to respiratory diseases, cardiovascular issues, and reduced quality of life, especially among vulnerable populations like children and the elderly.

---

###  Why is this Problem important :
 
Air pollution is a major issue because it directly impacts public health, causing respiratory and cardiovascular diseases, increasing healthcare costs, and reducing productivity. It also damages ecosystems, contributes to climate change, and disproportionately affects vulnerable communities, intensifying social inequalities. Addressing this problem is essential for improving urban livability, economic resilience, and environmental sustainability.


### How it impacts urban dev: 
 
Understanding and improving the Air Quality Index (AQI) is key to addressing these issues. Monitoring AQI helps policymakers implement effective regulations and allows citizens to make informed choices about their health and safety. Mitigating air pollution is essential for ensuring that cities can develop sustainably, promoting cleaner transportation, greener infrastructure, and stronger environmental policies to create more livable, equitable, and resilient urban spaces.

---

### Solution 

The primary problem this solution addresses is:

*Analyze and visualize air quality data* (focusing on PM2.5 levels) for a user-specified time range.

*Provide actionable health advisories* based on specific PM2.5 thresholds.

The solution effectively addresses this problem in the following ways:

**Date-Range-Based Filtering:**

The user specifies a custom date range, and the program dynamically filters the air quality data based on that range. This allows the user to focus on specific time periods that are of interest, making the solution flexible for analyzing historical air quality data.

**Average PM2.5 Calculation:**

The solution calculates the daily average PM2.5 values for each date within the selected range. The use of aggregation by date provides a summary view of how the air quality fluctuated during the time span, simplifying the complex, row-level data for easier understanding.

**Visualization:**

The program visualizes the PM2.5 data using a line chart, which enables the user to see trends and spikes in air pollution levels over time. This visual feedback helps the user quickly assess air quality variations and pinpoint concerning periods (e.g., rapid increases in PM2.5 levels).

**Health Advisories:**

The Tkinter-based alert system provides detailed health guidance based on predefined PM2.5 levels. This feature is not just informational but prescriptive, offering advice for users to take necessary precautions when air quality worsens.

The inclusion of detailed advisories for various PM2.5 thresholds (from 50 to 300) ensures that the user is informed about risks as air quality worsens, and they can take preventive or corrective measures accordingly.

### Core Innovation

The core innovation of this solution lies in its integration of data analysis, visualization, and actionable health guidance in a seamless workflow.

**User-Centered Data Filtering:**

By allowing users to input a specific date range, the solution is highly personalized. The flexibility of user-defined input ensures that the system adapts to different user needs, such as analyzing data from a specific month, season, or event.

**Real-Time Feedback with Visual and Textual Information:**

Combining data visualization with automated health alerts is the key innovation here. The user not only sees the historical data but also receives real-time contextual information about what the data means for their health and actions they should take. Most traditional solutions stop at visualizing data but do not translate that into practical user advice.

**Accessibility:**

The solution uses simple tools like matplotlib for plotting and tkinter for pop-up messages, making it lightweight and easy to use on a variety of systems. The use of Tkinter for pop-up alerts adds a layer of user engagement beyond the command-line interface, creating a more interactive and intuitive experience.

**Health-Centric Focus:**

Unlike typical data analysis solutions that focus solely on numbers and trends, this solution adds a health-centric layer by providing detailed, actionable advisories. These advisories give the solution a real-world impact beyond data analysis, helping users make informed decisions based on the air quality data.

In summary, the solution’s core innovation is its ability to merge data-driven insights (PM2.5 averages) with actionable health warnings in an accessible, user-friendly format. This combination of dynamic data analysis, visualization, and user alerts makes it highly useful for real-world air quality monitoring and health safety purposes.


### Technical Architecture
 

- AI algo/models used : 
	 - **Linear Regression** - Linear Regression is a simple and interpretable algorithm that fits a line through the data points and makes predictions based on the relationship between the input features and the target AQI value.
	 
	 - **Lasso Regression** - Lasso (Least Absolute Shrinkage and Selection Operator) is a type of linear regression that adds a regularization term to the loss function to penalize large coefficients, making the model less sensitive to noise and overfitting.
	
	 - **Decision Tree Regressor** - A Decision Tree Regressor splits the data based on certain feature thresholds, making predictions by averaging the AQI values in the leaf nodes.
	 
	 - **KNN Regressor** - K-Nearest Neighbors (KNN) is a simple algorithm that predicts the AQI value based on the average of the k-nearest data points in the feature space.
	 
	 - **RandomForestRegressor** - Random Forest is an ensemble of decision trees that improves prediction by averaging multiple trees, reducing overfitting, and improving accuracy.
	 
	 - **Xgboost Regressor** - XGBoost is a gradient boosting algorithm that builds trees sequentially and optimizes the model for better performance.
	 
	 - **Huperparameter Tuning** - Hyperparameter tuning involves finding the best set of parameters (e.g., number of trees, learning rate, max depth) for models like Random Forest, XGBoost, and KNN using techniques like Grid Search or Randomized Search.
	
	- **ANN (Artificial Neural Network)** - ANN can capture complex patterns in the data using multiple layers of neurons. It is useful for making predictions when the relationship between the features and AQI is highly nonlinear.

- Datasets : 
	- [Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
	- [NCBI research paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10280551/)
- Programming Languages : 
	- Python
	- Markdown

- Software/tools used:
	- Visual Studio Code 
	- Github
	- Git
	- Intel based Laptops (Intel CPUs)
- Visual Diagram  

## Social, Ecological and Economic Impact 

- ### Sustainability : 
Our Air Quality Index (AQI) project promotes sustainability by addressing air pollution, which impacts environmental, social, and economic aspects of urban life. By monitoring and raising awareness of air quality, the project encourages reducing emissions, protecting ecosystems, and improving public health. It also contributes to climate change mitigation by targeting pollutants that are climate forcers. Additionally, the project promotes social equity by highlighting the disproportionate effects of pollution on vulnerable communities and supports sustainable urban planning through data-driven decisions that enhance livability and resilience in cities.

- ### Impact when scaled : 
	
	**Environmental Impact**
	- **Improved Air Quality**: Scaling our project will lead to better monitoring and management of pollution sources, reducing harmful emissions from vehicles, industries, and other contributors.
	- **Biodiversity Preservation**: Cleaner air will result in healthier ecosystems, as reduced pollution will lower the damage to plants, animals, and water bodies, supporting greater biodiversity.
	- **Climate Change Mitigation**: By reducing pollutants that also act as climate forcers (e.g., black carbon), our project will help mitigate the effects of climate change, contributing to lower global temperatures and more resilient urban environments.

	**Social Impact**
	- **Enhanced Public Health**: Improved air quality will decrease the prevalence of respiratory and cardiovascular diseases, leading to healthier populations and longer life expectancy, particularly benefiting vulnerable groups such as children and the elderly.
	- **Greater Public Awareness**: As more people become informed about air quality, they can take actions to reduce exposure to pollutants and advocate for cleaner environments, empowering communities.
	- **Social Equity**: Scaling our project will help address inequalities, as low-income and vulnerable communities often face the brunt of poor air quality. By targeting these areas, our project can ensure cleaner air for all.
 
	**Economic Impact**
	- **Reduced Healthcare Costs**: With fewer air pollution-related diseases, the burden on healthcare systems will decrease, leading to significant savings in medical expenses and greater productivity due to fewer sick days.
	- **Increased Productivity**: Healthier populations result in higher labor productivity and economic output. Cleaner air also improves overall livability, making cities more attractive to businesses and residents.
	- **Sustainable Urban Development**: Better air quality can lead to more sustainable infrastructure investments, such as in green transportation and energy-efficient buildings, fostering long-term economic growth and resilience.

## Integration 

### **1. Data Integration with Monitoring Stations**

Our project can enhance its accuracy by collaborating with local or national environmental agencies to access real-time data from existing air quality monitoring stations. Additionally, integrating Internet of Things (IoT) sensors allows for localized air quality data collection, providing more granular monitoring and predictions in specific areas.

### **2. Integration with Public Health Systems**

Connecting our project with public health databases can correlate air quality data with health outcomes, offering insights into the impacts of air pollution on community health. By integrating with public health alert systems, our project can notify citizens of poor air quality levels and provide health recommendations to enhance awareness and response.

### **3. Collaboration with Urban Planning Tools**

Our project can support urban planning by integrating air quality predictions with urban development systems, enabling planners to make informed decisions about zoning and infrastructure. Aligning with smart city initiatives allows for the use of data analytics to promote urban sustainability and resilience.

### **4. Integration with Educational Platforms**

Collaborating with schools and educational institutions can foster community engagement and awareness about air quality issues. Developing interactive dashboards will make air quality data more accessible and actionable for residents, encouraging participation in monitoring and advocacy efforts.

### **5. Policy and Regulation Support**

Our project can provide policymakers with crucial data on air quality trends, aiding in the development of regulations aimed at reducing pollution and promoting public health. By integrating with systems that assess environmental impacts, our project can help evaluate the effectiveness of air quality interventions and inform future strategies.

## Demo 

[Youtube Video](https://youtu.be/cU2qOUErZMc)

[Github repository](https://github.com/SUDIPTO1903/AIR-QUALITY-INDEX)

## Future Potential and Scalability

### **Market Opportunities**

1. **Health and Wellness Sector:** Growing awareness of air pollution's health impacts creates demand for real-time air quality data, appealing to health-conscious consumers and families.
    
2. **Smart City Solutions:** Our project can integrate with smart city initiatives, offering valuable data analytics for urban management and sustainability efforts.
    
3. **Environmental Compliance and Regulation:** Businesses need tools to monitor emissions and comply with regulations, positioning our project as a useful resource for various industries.
    
4. **Educational Institutions:** Our project can be marketed to schools and universities as a resource for environmental education, engaging students in learning about air quality issues.
    
5. **Mobile and Web Applications:** Developing applications that provide personalized air quality data taps into trends in health monitoring and environmental awareness, attracting a broad user base.
    
### **Social Opportunities**

1. **Community Health Awareness:** By providing local air quality information, our project empowers communities to make informed health decisions, especially for vulnerable populations.
    
2. **Advocacy for Policy Change:** Highlighting air quality impacts can drive community advocacy for stricter regulations and better environmental policies.
    
3. **Public Engagement and Education:** Our project can promote understanding of air quality issues, encouraging citizen participation in environmental initiatives.
    
4. **Social Equity:** Focusing on disparities in air quality allows our project to advocate for equitable access to clean air, particularly for low-income communities.
    
5. **Collaborative Initiatives:** Our project can foster partnerships with local organizations and government agencies to implement community-based air quality improvement programs, enhancing social cohesion.>>
