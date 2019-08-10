import numpy as np
from math import tan, sin, cos

class Drone:

    def __init__(self, t0, tf, sim_step, drag,
                 initial_constraints_x, final_constraints_x,
                 initial_constraints_y, final_constraints_y,
                 initial_constraints_z, final_constraints_z,
                 initial_constraints_psi, final_constraints_psi):
        self.t0 = t0
        self.tf = tf
        self.sim_step = sim_step
        self.states = np.zeros((int((tf-t0)/sim_step), 9))
        self.x_ref = np.zeros((int((tf-t0)/sim_step), 1))
        self.y_ref = np.zeros((int((tf-t0)/sim_step), 1))
        self.z_ref = np.zeros((int((tf-t0)/sim_step), 1))
        self.psi_ref = np.zeros((int((tf-t0)/sim_step), 1))
        self.t = np.zeros((int((tf-t0)/sim_step), 1))
        self.states[0,:] = np.array([initial_constraints_x[0],initial_constraints_y[0],initial_constraints_z[0],
                                     initial_constraints_x[1],initial_constraints_y[1],initial_constraints_z[1],
                                     0, 0, initial_constraints_psi[0]])
        self.D = drag
        self.dx = drag[0,0]
        self.dy = drag[1,1]
        self.dz = drag[2,2]
        self.g = 9.8
        self.c_p_x,self.c_v_x,self.c_a_x, self.c_j_x = self.generate_polynomial_trajectory(initial_constraints_x,final_constraints_x,t0,tf)
        self.c_p_y,self.c_v_y,self.c_a_y, self.c_j_y = self.generate_polynomial_trajectory(initial_constraints_y,final_constraints_y,t0,tf)
        self.c_p_z,self.c_v_z,self.c_a_z, self.c_j_z = self.generate_polynomial_trajectory(initial_constraints_z,final_constraints_z,t0,tf)
        self.c_p_psi,self.c_v_psi,self.c_a_psi, self.c_j_psi = self.generate_polynomial_trajectory(initial_constraints_psi,final_constraints_psi,t0,tf)

    def generate_polynomial_trajectory(self,initial_constraints,final_constraints,t0,tf):
        x0 = initial_constraints[0]
        v0 = initial_constraints[1]
        a0 = initial_constraints[2]

        xf = final_constraints[0]
        vf = final_constraints[1]
        af = final_constraints[2]

        A = np.array([[1,              t0 ** 1,      t0 ** 2,        t0 ** 3,     t0 ** 4,     t0 ** 5],
                      [1,              tf ** 1,      tf ** 2,        tf ** 3,     tf ** 4,     tf ** 5],
                      [0,              1,             2 * t0 ** 1,    3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
                      [0,              1,             2 * tf ** 1,    3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                      [0,              0,             2,              6 * t0 ** 1, 12 * t0 ** 2,20 * t0 ** 3],
                      [0,              0,             2,              6 * tf ** 1, 12 * tf ** 2,20 * tf ** 3]])

        b = np.array([[x0], [xf], [v0], [vf], [a0], [af]])

        c_p = np.linalg.solve(A,b)
        c_v = np.polynomial.polynomial.polyder(c_p)
        c_a = np.polynomial.polynomial.polyder(c_v)
        c_j = np.polynomial.polynomial.polyder(c_a)
        return (c_p,c_v,c_a,c_j)


    def drone_model(self,states,inputs):
        g = 9.8
        v_x = states[3]
        v_y = states[4]
        v_z = states[5]
        phi = states[6]
        theta = states[7]
        psi = states[8]
        p = inputs[0]
        q = inputs[1]
        r = inputs[2]
        T = inputs[3]

        R_d_angle = np.array([[1, tan(theta) * sin(phi),  tan(theta) * cos(phi)],
                              [0, cos(phi),               -sin(phi)],
                              [0, sin(phi) / cos(theta),  cos(phi) / cos(theta)]])

        R_E_B = np.array([[cos(theta) * cos(psi), cos(theta) * sin(psi), - sin(theta)],
                          [sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi),
                           sin(phi) * cos(theta)],
                          [cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi), cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi),
                           cos(phi) * cos(theta)]])

        R_B_E = np.transpose(R_E_B)

        dp = np.array([[v_x],[v_y],[v_z]])
        dv = np.array([[[0], [0],  [g]]])+R_B_E*np.array([[0],[0],[T]])+R_B_E*self.D*R_E_B*np.array([[v_x],[v_y],[v_z]])
        dPhi = R_d_angle*np.array([[p],[q],[r]])
        return np.array([[dp],[dv],[dPhi]])

    def run_simulation(self):
        for i in range(np.size(self.states,0)-1):
            if i == 0:
               self.t[i] = self.t0 + i*self.sim_step
               self.x_ref[i] = np.polyval(self.c_p_x,self.t[i])
               self.y_ref[i] = np.polyval(self.c_p_y,self.t[i])
               self.z_ref[i] = np.polyval(self.c_p_z,self.t[i])

            v = np.array([np.polyval(self.c_v_x,self.t[i]),np.polyval(self.c_v_y,self.t[i]),np.polyval(self.c_v_z,self.t[i])])

            a = np.array([np.polyval(self.c_a_x,self.t[i]),np.polyval(self.c_a_y,self.t[i]),np.polyval(self.c_a_z,self.t[i])])

            jerk = np.array([np.polyval(self.c_j_x,self.t[i]), np.polyval(self.c_j_y,self.t[i]), np.polyval(self.c_j_z,self.t[i])])
            psi = np.polyval(self.c_p_psi,self.t[i])
            dPsi = np.polyval(self.c_v_psi,self.t[i])

            x_c = np.array([[cos(psi)], [sin(psi)], [0]])

            y_c = np.array([[-sin(psi)], [cos(psi)], [0]])

            z_w = np.array([[0], [0], [1]])
            alpha = a - self.g * z_w -self.dx * v
            beta = a - self.g * z_w -self.dy * v

            print(np.shape(alpha))

            x_b = np.cross(alpha,y_c,axis=0)/np.linalg.norm(np.cross(y_c,alpha,axis=0))
            y_b = np.cross(x_b,beta,axis=0)/np.linalg.norm(np.cross(beta,x_b,axis=0))
            z_b = np.cross(x_b,y_b,axis=0)

            print(np.shape(y_c))
            print(np.shape(z_b))
            print(np.dot(y_c.T,z_b))
            T = np.dot(z_b.T,a-self.g*z_w-self.dz*v)

            A =np.array([
                [0, np.ndarray.item(T+(self.dx-self.dx)*np.dot(z_b.T,v)), np.ndarray.item((self.dx-self.dy)*(np.dot(z_b.T,v)))],
                [np.ndarray.item(T-(self.dy-self.dz)*(np.dot(z_b.T,v))), 0, np.ndarray.item(-(self.dx-self.dy)*(np.dot(x_b.T,v)))],
                [0, np.ndarray.item(-np.dot(y_c.T,z_b)),np.linalg.norm(np.cross(y_c,z_b,axis=0))]])

            b = np.array([
                [np.ndarray.item(np.dot(x_b.T,jerk)-self.dx*np.dot(x_b.T,a))],
                [np.ndarray.item(-np.dot(y_b.T,jerk)+self.dy*np.dot(y_b.T,a))],
                [np.ndarray.item(dPsi*np.dot(x_c.T,x_b))]
            ])
            angular_rate = np.linalg.solve(A,b)
            inputs = np.concatenate((angular_rate,[T]),axis=None)
            self.states[i+1,:] = self.states[i,:] + self.sim_step*np.transpose(self.drone_model(self.states[i,:],inputs))

            self.t[i+1] = self.t0 + (i+1)*self.sim_step
            self.x_ref[i+1] = np.polyval(self.c_p_x,self.t[i+1])
            self.y_ref[i+1] = np.polyval(self.c_p_y,self.t[i+1])
            self.z_ref[i+1] = np.polyval(self.c_p_z,self.t[i+1])


