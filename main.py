import drone_differential_flatness as drone_df
import numpy as np

Drag = np.array([[-0.5,0,0],
                [0,-0.5,0],
                [0,0,-0.5]])
t0 = 0
tf = 10
sim_step = 1/500
initial_constraints_x = np.array([0,0,0])
final_constraints_x = np.array([5,0,0])
initial_constraints_y = np.array([0,0,0])
final_constraints_y = np.array([3,0,0])
initial_constraints_z = np.array([-1.5,0,0])
final_constraints_z = np.array([-2.5,0,0])
initial_constraints_psi = np.array([0,0,0])
final_constraints_psi = np.array([0,0,0])


drone = drone_df.Drone(t0,tf,sim_step,Drag,initial_constraints_x,final_constraints_x,
                       initial_constraints_y,final_constraints_y,
                       initial_constraints_z,final_constraints_z,
                       initial_constraints_psi,final_constraints_psi)

drone.run_simulation()


print("done")

