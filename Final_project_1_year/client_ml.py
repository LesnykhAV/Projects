import requests

url = "http://127.0.0.1:5000/predict"

sample_data = {
    "status": "for sale",
    "propertyType": "house",
    "city": "San Francisco",
    "zipcode": "94107",
    "state": "CA",
    "baths": 2,
    "sqft": 1200,
    "beds": 3,
    "stories": 1,
    "heating": 1,
    "cooling": 1,
    "parking": 1,
    "lotsize": 2000,
    "house_age": 15,
    "was_remodeled": 0,
    "remodeled_age": 0,
    "average_school_rating": 8,
    "average_school_distance": 2,
    "school_count_in_area": 3
}

response = requests.post(url, json=sample_data)
print(response.json())
