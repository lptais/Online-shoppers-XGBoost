import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
import json

# define model, version and project info
version = "v1"
project = "ml-deployment1"
model = "xgboost_tuned"
region = "europe-west3"
features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated',
            'BounceRates', 'PageValues', 'SpecialDay', 'Month', 'OperatingSystems',
           'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']

def predict_json(request):
    request_json = request.get_json()
    """Send json data to a deployed model for prediction.
    Args:
        request: A json with feature names and its values
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    print(request.args)
    instance_list = []
    for feature in features:
        if request.args and feature in request.args:
            value = request.args.get(feature)
            instance_list.append(value)
        elif request_json and feature in request_json:
            value = request_json[feature]
            instance_list.append(value)
    instances = [instance_list]
    print('instances', instances)

    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)
    service = googleapiclient.discovery.build(
        'ml', 'v1', client_options=client_options)
    name = 'projects/{}/models/{}'.format(project, model)
    if version is not None:
        name += '/versions/{}'.format(version)
    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    print('response', response)
    if 'error' in response:
        raise RuntimeError(response['error'])
    return json.dumps(response['predictions'])
