import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from matplotlib.colors import LogNorm
import h5py

class Functions():
    def __init__(self,delta,grid_size):
        self.delta = delta
        self.grid_size = grid_size

    def gradient(self,field):
        grad_x = np.zeros((self.grid_size,self.grid_size,self.grid_size))
        grad_y = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        grad_z = np.zeros((self.grid_size, self.grid_size, self.grid_size))

        grad_x[1:-1,1:-1,1:-1] = (field[2::,1:-1,1:-1]-field[0:-2,1:-1,1:-1])/(2*self.delta)
        grad_y[1:-1,1:-1,1:-1] = (field[1:-1,2::,1:-1]-field[1:-1,0:-2,1:-1])/(2*self.delta)
        grad_z[1:-1,1:-1,1:-1] = (field[1:-1,1:-1,2::]-field[1:-1,1:-1,0:-2])/(2*self.delta)
        grad = np.array([grad_x,grad_y,grad_z])
        return grad

    def divergence(self,field):
        grad_x = self.gradient(field[0, :, :, :])
        grad_y = self.gradient(field[1, :, :, :])
        grad_z = self.gradient(field[2, :, :, :])
        componentDiv = grad_x[0]+grad_y[1]+grad_z[2]
        return componentDiv

    def laplac(self,field):
        componentLap = np.sum(self.derivativeo2(field),axis=0)
        return componentLap

    def derivativeo2(self,field):
        deriv_x = np.zeros((3, self.grid_size, self.grid_size, self.grid_size))
        deriv_y = np.zeros((3, self.grid_size, self.grid_size, self.grid_size))
        deriv_z = np.zeros((3, self.grid_size, self.grid_size, self.grid_size))

        deriv_x[:,1:-1,1:-1,1:-1] = (field[:,2::,1:-1,1:-1]-2*field[:,1:-1,1:-1,1:-1]+field[:,0:-2,1:-1,1:-1])/self.delta**2
        deriv_y[:,1:-1,1:-1,1:-1] = (field[:,1:-1,2::,1:-1]-2*field[:,1:-1,1:-1,1:-1]+field[:,1:-1,0:-2,1:-1])/self.delta**2
        deriv_z[:,1:-1,1:-1,1:-1] = (field[:,1:-1,1:-1,2::]-2*field[:,1:-1,1:-1,1:-1]+field[:,1:-1,1:-1,0:-2])/self.delta**2
        derivo2 = np.array([deriv_x,deriv_y,deriv_z])
        return derivo2

    def grad_div(self, field):
        grad_div_x = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        grad_div_y = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        grad_div_z = np.zeros((self.grid_size, self.grid_size, self.grid_size))

        xx_A_x = (field[0,2::,1:-1,1:-1]-2*field[0,1:-1,1:-1,1:-1]+field[0,0:-2,1:-1,1:-1])/self.delta**2
        xy_A_y = (field[1,2::,2::,1:-1]-field[1,2::,0:-2,1:-1]-field[1,0:-2,2::,1:-1]+field[1,0:-2,0:-2,1:-1])
        xz_A_z = (field[2,2::,1:-1,2::]-field[2,2::,1:-1,0:-2]-field[2,0:-2,1:-1,2::]+field[2,0:-2,1:-1,0:-2])
        grad_div_x[1:-1,1:-1,1:-1] = self.delta**2*(xx_A_x+0.25*xy_A_y+0.25*xz_A_z)

        yx_A_x = (field[0,2::,2::,1:-1]-field[0,2::,0:-2,1:-1]-field[0,0:-2,2::,1:-1]+field[0,0:-2,0:-2,1:-1])
        yy_A_y = (field[1,1:-1,2::,1:-1]-2*field[1,1:-1,1:-1,1:-1]+field[1,1:-1,0:-2,1:-1])
        yz_A_z = (field[2,1:-1,2::,2::]-field[2,1:-1,2::,0:-2]-field[2,1:-1,0:-2,2::]+field[2,1:-1,0:-2,0:-2])
        grad_div_y[1:-1,1:-1,1:-1] = self.delta**2*(0.25*yx_A_x + yy_A_y + 0.25*yz_A_z)

        zx_A_x = (field[0,2::,1:-1,2::]-field[0,2::,1:-1,0:-2]-field[0,0:-2,1:-1,2::]+field[0,0:-2,1:-1,0:-2])
        zy_A_y = (field[1,1:-1,2::,2::]-field[1,1:-1,2::,0:-2]-field[1,1:-1,0:-2,2::]+field[1,1:-1,0:-2,0:-2])
        zz_A_z = (field[2,1:-1,1:-1,2::]-2*field[2,1:-1,1:-1,1:-1]+field[2,1:-1,1:-1,0:-2])
        grad_div_z[1:-1,1:-1,1:-1] = self.delta**2*(0.25*zx_A_x + 0.25*zy_A_y + zz_A_z)

        return np.array([grad_div_x,grad_div_y,grad_div_z])

    def curl(self,field):
        curl_x = self.directionalDeriv(field, component=2, direction=1) - self.directionalDeriv(field, component=1,
                                                                                    direction=2)
        curl_y = self.directionalDeriv(field, component=0, direction=2) - self.directionalDeriv(field, component=2,
                                                                                    direction=0)
        curl_z = self.directionalDeriv(field, component=1, direction=0) - self.directionalDeriv(field, component=0,
                                                                                    direction=1)
        curl = np.array(curl_x,curl_y,curl_z)
        return curl

    def directionalDeriv(self,field_component,component=0,direction=0):

        if direction == 0:
            deriv_x = np.zeros((3,self.grid_size,self.grid_size,self.grid_size))
            deriv_x[1:-1,1:-1,1:-1] = (field_component[component,2::,1:-1,1:-1]-field_component[component,0:-2,1:-1,1:-1])/(2*self.delta)
            return deriv_x

        elif direction == 1:
            deriv_y = np.zeros((3, self.grid_size, self.grid_size, self.grid_size))
            deriv_y[1:-1,1:-1,1:-1] = (field_component[component,1:-1,2::,1:-1]-field_component[component,1:-1,0:-2,1:-1])/(2*self.delta)
            return deriv_y

        elif direction == 2:
            deriv_z = np.zeros((3,self.grid_size,self.grid_size,self.grid_size))
            deriv_z[1:-1,1:-1,1:-1] = (field_component[component,1:-1, 1:-1, 2::] - field_component[component, 1:-1, 1:-1, 0:-2]) / (2 * self.delta)
            return deriv_z
        else:
            print("direction is not within 0-2. direction=",direction)

class Maxwell(Functions):
    def __init__(self,x_space,n_grids,t_max):
        self.n = n_grids
        self.x_grid = x_space
        x_lenght = x_space[-1]-x_space[0]
        self.delta = x_lenght/(n_grids-2)
        Functions.__init__(self,self.delta,n_grids)
        self.E = np.zeros((3,self.n,self.n,self.n))+0.0001
        self.A = np.zeros((3,self.n,self.n,self.n))
        self.rho = np.zeros((3,self.n,self.n,self.n))
        self.j = np.zeros((3,self.n,self.n,self.n))
        self.phi = np.zeros((self.n,self.n,self.n))
        self.constraint = 0
        self.E_source = np.zeros((3,self.n,self.n,self.n))
        self.t_max = t_max
        self.t = 0

        self.r2 = (self.x_grid**2+self.x_grid[np.newaxis,:].T**2)[np.newaxis,:,:]+self.x_grid[np.newaxis,np.newaxis,:].T**2
        self.initialize()
        self.create_dataset()

    def initialize(self):
        self.source()
    def create_dataset(self):
        hfE = h5py.File('data\maxwell_E.h5','w')
        hfE.create_dataset('Ex', data=np.resize(self.E[0],(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfE.create_dataset('Ey', data=np.resize(self.E[1],(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfE.create_dataset('Ez', data=np.resize(self.E[2],(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfE.create_dataset('E_norm', data=np.resize(np.linalg.norm(self.E,axis=0),(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfE.close()

        hfA = h5py.File('data\maxwell_A.h5', 'w')
        hfA.create_dataset('Ax', data=np.resize(self.A[0],(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfA.create_dataset('Ay', data=np.resize(self.A[1],(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfA.create_dataset('Az', data=np.resize(self.A[2],(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfA.create_dataset('A_norm', data=np.resize(np.linalg.norm(self.A,axis=0),(1,self.n,self.n,self.n)),maxshape=(None,self.n,self.n,self.n),compression='gzip')
        hfA.close()

    def update_dataset(self):
        hfE = h5py.File('data\maxwell_E.h5', 'a')
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
        E_x, E_y, E_z = self.partial_deriv(self.E)
        A_x, A_y, A_z = self.partial_deriv(self.A)
        E_dot[:, 0, :, :] = -self.E[:, 0, :, :]
        E_dot[:, :, 0, :] = -self.E[:, :, 0, :]
        E_dot[:, :, :, 0] = -self.E[:, :, :, 0]

        E_dot[:, -1, :, :] = -self.E[:, -1, :, :]
        E_dot[:, :, -1, :] = -self.E[:, :, -1, :]
        E_dot[:, :, :, -1] = -self.E[:, :, :, -1]

        return E_dot, A_dot

    def equations(self,fields):
        self.E = fields[0]
        self.A = fields[1]
        self.source()
        E_dot = -self.laplac(self.A)+self.grad_div(self.A)-4*np.pi*self.j
        A_dot = -self.E-self.gradient(self.phi)
        E_dot, A_dot = self.boundary_condition(E_dot,A_dot)

        return [E_dot,A_dot]

    def update_field(self, fields, fields_dot, factor, dt):
        new_fields = []
        for var in range(0,2):
            field = fields[var]
            field_dot = fields_dot[var]
            new_field = field + factor*field_dot * dt
            new_fields.append(new_field)
        return new_fields


    def ICN(self,fields,dt):
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
        print("Integrating to time",t_max)
        print("Checking results after",t_check)
        time = 0
        while time + t_check <= self.t_max:
            delta_t = courant * self.delta
            t_fin = time + t_check
            fields = [self.E, self.A]
            print("Integrating from", time, "to",t_fin)
            while time < t_fin:
                if t_fin - time < delta_t:
                    delta_t = t_fin - time
                fields = self.ICN(fields, delta_t)
                time += delta_t
                self.t = time
            self.E, self.A = fields
            self.check_constraints()
            self.update_dataset()

def updatefig(i,img,n):
    if n == 1:
        img.set_array(E_norm[i,int(n_grids/2),1:-1,1:-1])
    elif n == 2:
        img.set_array(E_norm[i, 1:-1, int(n_grids/2), 1:-1])
    elif n == 3:
        img.set_array(E_norm[i, 1:-1, 1:-1, int(n_grids/2)])
    return [img]

n_grids = 50
x_grid = np.linspace(-5,5,n_grids)
t_max = 10
t_check = 0.1
courant = 0.5
CREATE_DATA = True
if CREATE_DATA == True:
    maxwell = Maxwell(x_grid,n_grids,t_max)
    maxwell.runner(t_check, courant)

hf = h5py.File('data\maxwell_E.h5')
E_norm = np.array(hf['E_norm'])
#Ex = np.array(hf['Ex'])
#Ey = np.array(hf['Ey'])
#Ez = np.array(hf['Ez'])

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(X,Y,Z,Ex[0],Ey[0],Ez[0])
#plt.show()
fig, ax = plt.subplots(figsize=(16,6))
#im1 = ax[0].imshow(E_norm[0,int(n_grids/2),1:-1,1:-1],cmap='plasma',norm=LogNorm(vmin=0.01,vmax=5))
#im2 = ax[1].imshow(E_norm[0,1:-1,int(n_grids/2),1:-1],cmap='plasma',norm=LogNorm(vmin=0.01,vmax=5))
#im3 = ax[2].imshow(E_norm[0,1:-1,1:-1,int(n_grids/2)],cmap='plasma',norm=LogNorm(vmin=0.01,vmax=5))
im3 = ax.imshow(E_norm[0,1:-1,1:-1,int(n_grids/2)],cmap='plasma',norm=LogNorm(vmin=0.01,vmax=5))

#ani1 = animation.FuncAnimation(fig, updatefig,fargs=[im1,1],frames=int(t_max/t_check-1),interval=25, blit=True)
#ani2 = animation.FuncAnimation(fig, updatefig,fargs=[im2,2],frames=int(t_max/t_check-1),interval=25, blit=True)
ani3 = animation.FuncAnimation(fig, updatefig,fargs=[im3,3],frames=int(t_max/t_check-1),interval=25, blit=True)
plt.show()