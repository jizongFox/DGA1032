import kaggle,os,subprocess

project_root_path = os.path.dirname(os.getcwd())
if not os.path.isdir(os.path.join(project_root_path,'dataset')):
    os.mkdir(os.path.join(project_root_path,'dataset'))
cmd = '"kaggle datasets download -d jizongpeng/acdc2dall --force -p %s"'%os.path.join(project_root_path,'dataset')
os.system('/bin/zsh -c '+ cmd)
