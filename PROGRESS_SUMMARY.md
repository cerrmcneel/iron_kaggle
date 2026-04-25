# Project Checkpoint: IronKaggle Redux

## Status: API Development Complete ✅
The local Flask API is fully functional and tested.

## What we built:
- **`app.py`**: A Flask server that handles POST requests to `/predict`.
- **Validation:** A robust loop that checks for 8 required fields and converts strings to numbers.
- **Date Handling:** Flexible parsing for `DD/MM/YYYY`, `YYYY-MM-DD`, etc.
- **Feature Engineering:** Automated extraction of Month, Day, Year, and Week of Year.
- **Model Compatibility:** Added dummy values for history features (`sales_lag`, `rolling_mean`) to satisfy the model requirements.
- **Business Logic:** Hard-coded logic to return 0 sales if the store is closed (`open: 0`).
- **Math:** Inverse log transform using `np.expm1`.

## How to Test Locally:
1. Run `python app.py`.
2. Use the following PowerShell command:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType "application/json" -Body '{"store_ID": 49, "day_of_week": 4, "date": "26/06/2014", "nb_customers_on_day": 1254, "open": 1, "promotion": 0, "state_holiday": "0", "school_holiday": 1}'
```

## Next Phase: Cloud Deployment
- [ ] Update `app.run(host='0.0.0.0')`.
- [ ] Create `Dockerfile`.
- [ ] Push to AWS ECR.
- [ ] Launch AWS EC2 and deploy.
- [ ] **Critical:** Delete AWS resources after testing!
