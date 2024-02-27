
import numpy as np
import geometry_helpers as gh
import icp


def generate_cloud(n_pts):
    points = np.random.rand(n_pts, 3) * 10
    return points


x_gt = np.float32([0, 20, 0, 0, 0, 0])

if __name__ == '__main__':
    cloud_fixed = generate_cloud(100)
    X_gt = gh.v2t(x_gt)
    #print('X_gt=\n', X_gt)
    cloud_moving = (X_gt[:3, :3] @ cloud_fixed.T + X_gt[:3, 3, np.newaxis]).T

    random_idx = np.random.choice(cloud_fixed.shape[0], 4, replace=False)
    print(random_idx)
    inliers_fixed = cloud_fixed[random_idx, :]
    inliers_moving = cloud_moving[random_idx, :]
    print(inliers_fixed)

    print(inliers_moving)

    #exit(0)

    X_estimate = icp.do_icp(cloud_fixed, cloud_moving)
    print('X_est=\n', X_estimate)
    total_error = 0.0
    for i in range(cloud_moving.shape[0]):
        error = np.linalg.norm(X_estimate[:3, :3].T @ (cloud_moving[i, :][np.newaxis].T -
                               X_estimate[:3, 3][np.newaxis].T) - cloud_fixed[i, :][np.newaxis].T)
        total_error += error
    print('total_error=', total_error)

    # Plot section
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(cloud_fixed[:, 0], cloud_fixed[:, 1], cloud_fixed[:, 2])
    # ax.scatter(cloud_moving[:, 0], cloud_moving[:, 1],
    #            cloud_moving[:, 2], marker='^')
    # plt.show()

    #exit(0)
