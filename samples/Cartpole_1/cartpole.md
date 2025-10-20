Create and Install a New External Project

By default, the template comes with the cartpole example. Let’s make a fresh project.

Open a new terminal.

Activate your Python virtual environment. If you already have it activated from installing Isaac Lab, or you’re using our Brev Launchable, you can skip this step.

If using conda:


conda activate env_isaaclab

Navigate to the Isaac Lab folder

For example,

cd ~/IsaacLab

Run the Isaac Lab script with the –new argument to create the template project:
Linux:


./isaaclab.sh --new

The template generator will ask you a few questions. In this command-line menu, use arrows to move and the spacebar to select an option, then press enter.

Task type:
External to create the project outside the Isaac Lab repo

Project Path: set to
/home/{your user name}/Cartpole
You can set this to a different location if you like, just remember where it is for later.

Project name:
Cartpole

Isaac Lab workflow:
Manager-based

Note
Use arrows to move selection up/down, space to select.

Direct-based workflow will structure the code differently. It’s a different way to formulate tasks in Isaac Lab that has its own benefits and tradeoffs. Read more here.

RL library:
skrl

RL algorithms for skrl:
PPO (Proximal Policy Optimization)

To install your external project, which essentially registers it with Isaac Lab, run the following from within the project folder for the Cartpole project. If you’re not sure where this directory is, see the output of the previous command.

Tip
Opening the external project folder in your IDE (code editing program) is a convenient way to have your terminal open to the correct location. Otherwise, make sure to navigate into the project diretory using the command below, changing the second parameter if your project is somewhere else.


python -m pip install -e source/Cartpole

To confirm our project was installed, run this command to list all installed environments.

python scripts/list_envs.py

Seeing our project listed in the output means it is installed and ready to go!


If you haven’t already, open the project folder in your IDE by going to File > Open Folder and choosing the project directory we used above. For example, ~/Cartpole/Cartpole