import os

root_dir = r"D:\Desktop\论文\BioFoundry\ocp"

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".py"):
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                new_content = content
                if "from torch_scatter" in content:
                    # Replace specific imports
                    lines = new_content.splitlines()
                    new_lines = []
                    for line in lines:
                        if "from torch_scatter" in line:
                             # Replace module name
                             # Handle 'from torch_scatter.utils import broadcast' -> 'from ocpmodels.common.utils import broadcast'
                             # Handle 'from torch_scatter import scatter' -> 'from ocpmodels.common.utils import scatter'
                             new_line = line.replace("torch_scatter.utils", "ocpmodels.common.utils")
                             new_line = new_line.replace("torch_scatter", "ocpmodels.common.utils")
                             new_lines.append(new_line)
                        else:
                             new_lines.append(line)
                    new_content = "\n".join(new_lines)
                
                if "from torch_cluster import radius_graph" in content:
                    lines = new_content.splitlines()
                    new_lines = []
                    for line in lines:
                         if "from torch_cluster import radius_graph" in line:
                             new_line = line.replace("torch_cluster", "ocpmodels.common.utils")
                             new_lines.append(new_line)
                         else:
                             new_lines.append(line)
                    new_content = "\n".join(new_lines)

                if "from torch_sparse import" in content:
                    lines = new_content.splitlines()
                    new_lines = []
                    for line in lines:
                         if "from torch_sparse import" in line:
                             new_line = line.replace("torch_sparse", "ocpmodels.common.utils")
                             new_lines.append(new_line)
                         else:
                             new_lines.append(line)
                    new_content = "\n".join(new_lines)

                if new_content != content:
                    print(f"Patching {filepath}")
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(new_content)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
