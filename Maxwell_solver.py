import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from matplotlib.colors import LogNorm
import h5py
from functions import Functions

class Maxwell(Functions):
    def __init__(self,x_space,n_grids,t_max):
        self.n = n_grids
        self.x_grid = x_space
        x_lenght = x_space[-1]-x_space[0]
        self.delta = x_lenght/(n_grids-2)
        Functions.__init__(self,self.delta,n_grids)
        self.E = np.zeros((3,self.n,self.n,self.n))+0.0001 #The small constant factor is to handle the logarithmic scale of the pyplot, but wont have any effect on the physics
        self.A = np.zeros((3,self.n,self.n,self.n))
        self.rho = np.zeros((3,self.n,self.n,self.n))
        self.j = np.zeros((3,self.n,self.n,self.n))
        self.phi = np.zeros((self.n,self.n,self.n))
        self.constraint = 0
        self.E_source = np.zeros((3,self.n,self.n,self.n))
        self.t_max = t_max
        self.t = 0

        self.r2 = (self.x_grid**2+self.x_grid[np.newaxis,:].T**2)[np.newaxis,:,:]+self.x_grid[np.newaxis,np.newaxis,:].T**2

        #Create the source points
        self.initialize()
        #Create the h5py dataset
        self.create_dataset()

    def initialize(self):
        self.source()
    def create_dataset(self):
        hfE = h5py.File('data\maxwell_E.h5','w')
        #Create a subset of maxwell_E that contains each component and the norm of the electric field.
        hfE.create_dataset('Ex', data=np.resize(self.E[0],(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfE.create_dataset('Ey', data=np.resize(self.E[1],(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfE.create_dataset('Ez', data=np.resize(self.E[2],(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfE.create_dataset('E_norm', data=np.resize(np.linalg.norm(self.E,axis=0),(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfE.close()

        hfA = h5py.File('data\maxwell_A.h5', 'w')
        #Create a subset of maxwell_A that contains each component and the norm of the gauge field.
        hfA.create_dataset('Ax', data=np.resize(self.A[0],(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfA.create_dataset('Ay', data=np.resize(self.A[1],(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfA.create_dataset('Az', data=np.resize(self.A[2],(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfA.create_dataset('A_norm', data=np.resize(np.linalg.norm(self.A,axis=0),(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfA.close()

    def update_dataset(self):
        hfE = h5py.File('data\maxwell_E.h5', 'a')
        #Append to the existing data
        new_shape = hfE['Ex'].shape[0]+1
        hfE['Ex'].resize(new_shape, axis=0)
        hfE['Ex'][-1:] = self.E[0]
        hfE['Ey'].resize(new_shape, axis=0)
        hfE['Ey'][-1:] = self.E[1]
        hfE['Ez'].resize(new_shape, axis=0)
        hfE['Ez'][-1:] = self.E[2]
        hfE['E_norm'].resize(new_shape, axis=0)
        hfE['E_norm'][-1:] = np.linalg.norm(self.E,axis=0)
        hfE.close()

        #Append to the existing data
        hfA = h5py.File('data\maxwell_A.h5', 'a')
        hfA['Ax'].resize(new_shape, axis=0)
        hfA['Ax'][-1:] = self.A[0]
        hfA['Ay'].resize(new_shape, axis=0)
        hfA['Ay'][-1:] = self.A[1]
        hfA['Az'].resize(new_shape, axis=0)
        hfA['Az'][-1:] = self.A[2]
        hfA['A_norm'].resize(new_shape, axis=0)
        hfA['A_norm'][-1:] = np.linalg.norm(self.A, axis=0)
        hfA.close()


    def source(self,static=True):
        radial = 1/(np.pi*self.r2)
        #Calculate the 1/r^2 electric dropoff from the pointsource
        #Define the source to either be stationary in time, or make it go in a circle.
        if static:
            self.E[:, int(self.n / 2) - 2:int(self.n / 2) + 1,
            int(self.n / 2) - 2:int(self.n / 2) + 1,
            int(self.n / 2) - 2:int(self.n / 2) + 1] = radial[int(self.n / 2) - 2:int(self.n / 2) + 1,
                                                       int(self.n / 2) - 2:int(self.n / 2) + 1,
                                                       int(self.n / 2) - 2:int(self.n / 2) + 1]
        else:
            self.E[:,int(self.n/2)-2+int(self.n/4*np.sin(1.25*2*np.pi*self.t/self.t_max)):int(self.n/2)+1+int(self.n/4*np.sin(1.25*2*np.pi*self.t/self.t_max)),
            int(self.n/2)-2+int(self.n/4*np.cos(1.25*2*np.pi*self.t/self.t_max)):int(self.n/2)+1+int(self.n/4*np.cos(1.25*2*np.pi*self.t/self.t_max)),
            int(self.n/2)-2:int(self.n/2)+1] = radial[int(self.n/2)-2:int(self.n/2)+1,
                                                          int(self.n/2)-2:int(self.n/2)+1,
                                                          int(self.n/2)-2:int(self.n/2)+1]


    def check_constraints(self):

        self.constraint = self.divergence(self.E)
        norm_c = np.linalg.norm(self.constraint)
        norm_c = np.sqrt(norm_c * self.delta ** 3)
        print(" Constraint violation at time",norm_c)
        return norm_c

    def boundary_conditions(self,E_dot,A_dot):
        #FUTURE WORK
        #Implement outgoing wave boundary condition
        pass

        return E_dot, A_dot

    def equations(self,fields):
        self.E = fields[0]
        self.A = fields[1]
        #Insert the source point
        self.source()
        #Calculate the temporal derivative in terms of maxwells equations
        E_dot = -self.laplac(self.A)+self.grad_div(self.A)-4*np.pi*self.j
        A_dot = -self.E-self.gradient(self.phi)
        #E_dot, A_dot = self.boundary_condition(E_dot,A_dot)
        return [E_dot,A_dot]

    def update_field(self, fields, fields_dot, factor, dt):
        new_fields = []
        #integrate the to t+dt.
        for var in range(0,2):
            field = fields[var]
            field_dot = fields_dot[var]
            new_field = field + factor*field_dot * dt
            new_fields.append(new_field)
        return new_fields


    def ICN(self,fields,dt):
        #Iterative Crank-Nicholson Scheme
        #step 1
        fields_dot = self.equations(fields)
        new_fields = self.update_field(fields, fields_dot, 0.5, dt)

        # step 2
        temp_fields = self.update_field(fields, fields_dot, 1, dt)
        fields_dot = self.equations(temp_fields)

        #step 3
        temp_fields = self.update_field(fields,fields_dot, 1 ,dt)
        fields_dot = self.equations(temp_fields)

        #step 4
        new_fields = self.update_field(new_fields, fields_dot, 0.5, dt)
        return new_fields


    def runner(self,t_check, courant):
        #Function that handles the updating and saving of data
        print("Integrating to time",self.t_max)
        print("Checking results after",t_check)
        time = 0
        #Integrate form t to t_max
        while time + t_check <= self.t_max:
            #Calculate the dt determined by the Courant-Friedrich-Lwey Condition
            delta_t = courant * self.delta
            t_fin = time + t_check
            fields = [self.E, self.A]
            print("Integrating from", time, "to",t_fin)
            while time < t_fin:
                if t_fin - time < delta_t:
                    #if the remaining time is less than dt define a new dt
                    delta_t = t_fin - time
                #Evolve the fields
                fields = self.ICN(fields, delta_t)
                time += delta_t
                self.t = time
            #Update the current fields
            self.E, self.A = fields
            #Check constraints
            self.check_constraints()
            #Save the current fields to the datasets.
            self.update_dataset()

def updatefig(i,data,n_grids,img,n):
    #Function to animate the pyplot figure.
    if n == 1:
        #slice of yz plane
        img.set_array(data[i,int(n_grids/2),1:-1,1:-1])
    elif n == 2:
        #slice of xz plane
        img.set_array(data[i, 1:-1, int(data/2), 1:-1])
    elif n == 3:
        #slice of xy plane
        img.set_array(data[i, 1:-1, 1:-1, int(data/2)])
    return [img]

def main():
    #Choose gridspace: n_grids x n_grids x n_grids
    n_grids = 50
    x_grid = np.linspace(-5,5,n_grids)

    #Choose how units of time to integrate up to.
    t_max = 10
    #integrate in chunks of t_check.
    t_check = 0.1
    #courant factor used for the Iterative Crank-Nicholson scheme
    courant = 0.5

    #Create new data or preview the already generated data
    CREATE_DATA = True
    if CREATE_DATA == True:
        maxwell = Maxwell(x_grid,n_grids,t_max)
        maxwell.runner(t_check, courant)

    #load the electric field data. Alternatively maxwell_A gives to get the gauge field.
    hf = h5py.File('data\maxwell_E.h5')
    E_norm = np.array(hf['E_norm'])

#--- Other options include -----
    #Ex = np.array(hf['Ex'])
    #Ey = np.array(hf['Ey'])
    #Ez = np.array(hf['Ez'])
#-------------------------------

    #fig = plt.figure()
    fig, ax = plt.subplots(figsize=(16,6))
    #View an animated figure of the elctric field on a slice of the xy plane Logscaled.
    im3 = ax.imshow(E_norm[0,1:-1,1:-1,int(n_grids/2)],cmap='plasma',norm=LogNorm(vmin=0.01,vmax=5))
    ani3 = animation.FuncAnimation(fig, updatefig,fargs=[E_norm,n_grids,im3,3],frames=int(t_max/t_check-1),interval=25, blit=True)
    plt.show()

if __name__ == '__main__':
    main()
