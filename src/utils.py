import boto3
import pickle
import json


def convert_list_to_str(lst):
    string = (
        str(lst).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    )
    if string == "None":
        return None
    return string


def convert_str_to_list(string):
    if string is None:
        return None
    string = string.replace(" ", "")
    if len(string) > 0 and string[-1] == ",":
        string = string[:-1]
    return_list = string.split(",")
    return_list.sort()
    return return_list


def get_indexed_agents():
    s3 = boto3.client("s3")
    agents = s3.get_object(
        Bucket="sagemaker-us-east-2-534295958235",
        Key=f"gitbot/agents.pkl",
    )["Body"].read()
    agents = pickle.loads(agents)
    return agents


def is_agent_indexed(agents, agent_name):
    if agent_name in agents:
        return True
    else:
        return False


def get_model_id(model_name, model_display_names, model_ids):
    for model in model_display_names:
        if model == model_name:
            model_id = model_ids[model_name.name].value
            return model_id


def save_indexed_agents(agents):
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket="sagemaker-us-east-2-534295958235",
        Key=f"gitbot/agents.pkl",
        Body=pickle.dumps(agents),
    )


def save_config(agent_name, config):
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket="sagemaker-us-east-2-534295958235",
        Key=f"gitbot/{agent_name}/config.json",
        Body=json.dumps(config),
    )


def delete_s3_agent_contents(agent_name):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("sagemaker-us-east-2-534295958235")
    for obj in bucket.objects.filter(
        Prefix=f"gitbot/{agent_name}"
    ):
        obj.delete()
