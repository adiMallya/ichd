## Predict API

Backend API for the Intracraneal Haemorrhage Detection and Classification. Use it to make a prediction on a CT slice(PNG or JPG)

***Endpoint:***

```bash
Method: POST
URL: {{URL}}/predict_api
```

The API uses direct upload method, to accept a image file as the input to return a response like this:

```json
{
  "predictions": ["any", "subdural"],
  "pred_probas": [
    "any: 95.0",
    "subdural: 91.0",
    "intracraneal: 0.0",
    "intraparenchymal: 0.0",
    "subarachnoid: 0.0",
    "epidural: 0.0",
  ],
}
```