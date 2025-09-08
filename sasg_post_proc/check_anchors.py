from tools import get_sasg, get_cycle_dirs, get_sasg_zero_bounds, get_test_points
import numpy as np
import sys
def check_anchors(base_run_dir):
    cycle_dirs = get_cycle_dirs(base_run_dir)
    print(cycle_dirs[-1])
    sasg = get_sasg(cycle_dirs[-1])
    sasg_orig = get_sasg(cycle_dirs[-1])
    sasg_zero = get_sasg_zero_bounds(cycle_dirs[-1])
    print('adding boundary anchors')
    print('NUM POINTS BEFORE:',sasg.gridStorage.getSize())
    sasg.add_boundary_anchors(level=1)
    print('NUM POINTS AFTER:',sasg.gridStorage.getSize())

    print('NUM POINTS IN SASG ZERO:',sasg_zero.gridStorage.getSize())

    
    print(list(sasg.anchor_boundary_points.keys())[0:10])
    
    # num_anchor_points = 0
    # for i in range(sasg.gridStorage.getSize()):
    #     gp = sasg.gridStorage.getPoint(i)
    #     unit_point = ()
    #     for j in range(sasg.dim):
    #         unit_point = unit_point + (gp.getStandardCoordinate(j),)
    #     box_point = sasg.point_transform_unit2box(unit_point)
        
    #     if box_point in sasg.anchor_boundary_points.keys():
    #         num_anchor_points += 1            
        
    # print('NUM ANCHOR POINTS:',num_anchor_points)
    # print('SHOULD BE:',len(sasg.anchor_boundary_points))
    
    print('should be 0',np.sum(np.array(list(sasg.anchor_boundary_points.values()))))
    
    # prediction at anchor points
    anchor_points = list(sasg.anchor_boundary_points.keys())
    print('anchor prediction:',sasg.surrogate_predict(anchor_points, n_jobs=0))
    print('sasg zero prediction:',sasg_zero.surrogate_predict(anchor_points, n_jobs=0))
    
    # compute train_error
    def compute_train_error(sasg, train_points, train_values):
        # for tp in train_points:
            # if tp not in sasg.train:
            #     raise ValueError(f'not in train, {tp}')
        train_predict = sasg.surrogate_predict(train_points, n_jobs=0)
        residual = (np.abs(train_predict - train_values))

        mse = np.mean(residual**2)
        return mse
    
    train_points, train_values = list(sasg.train.keys()), list(sasg.train.values())
    train_points, train_values = train_points, train_values
    print('origional sasg train mse', compute_train_error(sasg_orig,train_points, train_values))
    print('anchor sasg train mse', compute_train_error(sasg,train_points, train_values))
    print('zero boundary sasg train mse', compute_train_error(sasg_zero,train_points, train_values))

    # mse for boundary
    print('for boundary, assuming true is 0')
    print('origional sasg mse', compute_train_error(sasg_orig, anchor_points, np.zeros(len(anchor_points))))
    print('anchor sasg mse', compute_train_error(sasg,anchor_points, np.zeros(len(anchor_points))))
    print('zero boundary sasg mse', compute_train_error(sasg_zero,anchor_points, np.zeros(len(anchor_points))))

    # mse for test set
    print('for test set')
    test_dir = '/scratch/project_2007848/DANIEL/data_store/MMMG_sobolseq_testset_for_zero'
    x_test, y_test = get_test_points(test_dir)
    print('origional sasg mse', compute_train_error(sasg_orig, x_test, y_test))
    print('anchor sasg mse', compute_train_error(sasg,x_test, y_test))
    print('zero boundary sasg mse', compute_train_error(sasg_zero,x_test, y_test))

if __name__ == '__main__':
    _, base_run_dir = sys.argv
    
    check_anchors(base_run_dir)
    