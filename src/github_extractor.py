from github import Github
import os

# Authentication is defined via github.Auth
from github import Auth

from typing import Union

import boto3
import os, shutil
from botocore.exceptions import ClientError


def build_txt_files(
    repos: Union[str, list],
    include_branches,
    include_folders,
    exclude_folders,
    documentation_folder_path,
    include_file_types,
    exclude_file_types,
    auth_token = None,
    files_dest_dir: str = "./tmp/",
):
    if documentation_folder_path is not None and documentation_folder_path[-1] != "/":
        documentation_folder_path += "/"
        
    # using an access token
    if auth_token is None:
        auth = Auth.Token(auth_token)

    if not os.path.exists(files_dest_dir):
        os.makedirs(files_dest_dir)
    else:
        for filename in os.listdir(files_dest_dir):
            file_path = os.path.join(files_dest_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

    def traverse_tree(tree, repo, path = ""):
        for element in tree.tree:
            if element.type == "blob":
                # If file is of type we want to exclude, skip it
                if exclude_file_types is not None and any(
                    file_type in os.path.join(rpo, path, element.path) for file_type in exclude_file_types
                ):
                    continue
                # If file is of type we want to include, add it to the list
                elif any(
                    file_type in os.path.join(repo, path, element.path) for file_type in include_file_types
                ):
                    # If file is in a folder we want to exclude, skip it
                    if exclude_folders is not None and any(
                        exclude_folder in os.path.join(repo, path, element.path)
                        for exclude_folder in exclude_folders
                    ):
                        continue
                    else:
                        # If we have specified folders to include, only include files in those folders
                        if include_folders is not None:
                            if any(
                                include_folder in os.path.join(repo, path, element.path)
                                for include_folder in include_folders
                            ):
                                files.append(os.path.join(path, element.path))
                        else:
                            files.append(os.path.join(path, element.path))
            elif element.type == "tree":
                sub_tree = repository.get_git_tree(element.sha)
                traverse_tree(sub_tree, repo, os.path.join(path, element.path))

    if isinstance(repos, str):
        repos = [repos]

    print(repos)
    for repo in repos:
        substring = repo.split("https://github.com/")[1]
        org, repo = substring.split("/")[0], substring.split("/")[1]

        # Github Enterprise with custom hostname
        if auth_token is None:
            g = Github(base_url="https://github.com/api/v3")
        else:
            g = Github(auth=auth, base_url="https://github.com/api/v3")
        repository = g.get_organization(org).get_repo(repo)

        files_dict = {}
        for branch_name in include_branches:
            try:
                branch = repository.get_branch(branch_name)
                print("Pulling from repo:", repository, "from branch:", branch_name)
                files = list()
                traverse_tree(branch.commit.commit.tree, repo)
                files_dict[branch_name] = files
            except Exception as e:
                print("Error in pulling data from the repository", str(e))

        for key in files_dict:
            for file in files_dict[key]:
                file_content = repository.get_contents(file, ref=key)
                new_file_path = os.path.join(
                    files_dest_dir,
                    repo + "/" + file_content.path.split(".")[0] + ".txt",
                )
                folder_hierarchy = new_file_path.split("/")
                folder_hierarchy.pop(-1)
                rel_path = folder_hierarchy[0]
                for j in range(1, len(folder_hierarchy)):
                    rel_path = os.path.join(rel_path, folder_hierarchy[j])
                    if not os.path.exists(rel_path):
                        os.makedirs(rel_path)
                if documentation_folder_path in new_file_path:
                    url_path = new_file_path.split(documentation_folder_path)[1]
                    url_path = url_path.replace(".txt", "/")
                    url_path = f"https://github.com/pages/{org}/{repo}/" + url_path
                    with open(new_file_path, "w", encoding="utf-8") as f:
                        # This line was edited
                        f.write(
                            "This file contains information regarding"
                            + file_content.path.split(".")[0]
                            + "This doc is hosted on a documentation site and is available at:"
                            + url_path 
                            + "\n\n Start of contents:\n\n"
                            + file_content.decoded_content.decode("utf-8")
                        )
                else:
                    with open(new_file_path, "w", encoding="utf-8") as f:
                        f.write(
                            "This file contains information regarding"
                            + file_content.path.split(".")[0]
                            + "\n\n Start of contents:\n\n"
                            + file_content.decoded_content.decode("utf-8")
                        )
