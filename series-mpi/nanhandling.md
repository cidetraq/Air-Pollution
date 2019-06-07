Need to reintegrate this into the pipeline using numpy after sequencing / feature generation

```
if self.masknan is not None:
    s = input['AQS_Code']
    input[input.isnull().any(axis=1)] = 1000
    input['AQS_Code'] = s
elif self.fillnan is not None:
    input.fillna(fillnan, inplace=True)
```
