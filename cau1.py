import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as solvers

app = Flask(__name__)

@app.route('/')
def index():
    # Hard margin SVM logic

    # 3 data points
    x = np.array([[1., 3.], [2., 2.], [1., 1.]])
    y = np.array([[1.], [1.], [-1.]])

    # Calculate H matrix
    H = np.dot(y, y.T) * np.dot(x, x.T)

    # Construct the matrices required for QP in standard form
    n = x.shape[0]
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((n, 1)))
    G = cvxopt_matrix(-np.eye(n))
    h = cvxopt_matrix(np.zeros(n))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))


    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10


    # Perform QP
    sol = solvers.qp(P, q, G, h, A, b)
    lamb = np.array(sol['x'])

    # Calculate w and b
    w = np.sum(lamb * y * x, axis=0)

    sv_idx = np.where(lamb > 1e-5)[0]

    sv_lamb = lamb[sv_idx]

    sv_x = x[sv_idx]
    sv_y = y[sv_idx].reshape(1, -1)
    b = np.mean(sv_y.flatten() - np.dot(sv_x, w))

    # Visualization
    plt.figure(figsize=(5, 5))
    color = ['red' if a == 1 else 'blue' for a in y]
    plt.scatter(x[:, 0], x[:, 1], s=200, c=color, alpha=0.7)
    plt.xlim(0, 4)
    plt.ylim(0, 4)

    # Decision boundary
    x1_dec = np.linspace(0, 4, 100)
    x2_dec = (-w[0] * x1_dec - b) / w[1]
    plt.plot(x1_dec, x2_dec, c='black', lw=1.0, label='decision boundary')

    # Positive & negative boundary
    w_norm = np.sqrt(np.sum(w ** 2))
    w_unit = w / w_norm
    half_margin = 1 / w_norm
    upper = np.vstack((x1_dec, x2_dec + half_margin * w_unit[1])).T
    lower = np.vstack((x1_dec, x2_dec - half_margin * w_unit[1])).T

    desired_points = x[np.isclose(lamb.flatten(), 1.0, atol=0.1) | np.isclose(lamb.flatten(), 0.0, atol=0.1)]

    x1_desired = desired_points[:, 0]
    x2_desired = desired_points[:, 1]

    slope = (x2_desired[1] - x2_desired[0]) / (x1_desired[1] - x1_desired[0])
    intercept = x2_desired[0] - slope * x1_desired[0]

    x2_adjusted = slope * x1_dec + intercept
    x_neg_start = np.array([0, 2])
    x_neg_end = np.array([2, (-w[0] * 3 - b) / w[1]])

    slope_neg = (x_neg_end[1] - x_neg_start[1]) / (x_neg_end[0] - x_neg_start[0])
    intercept_neg = x_neg_start[1] - slope_neg * x_neg_start[0]

    x2_neg_adjusted = slope_neg * x1_dec + intercept_neg


    plt.plot(x1_dec, x2_adjusted, '--', lw=1.0, label='positive boundary', color='#33CCFF')
    plt.plot(x1_dec, x2_neg_adjusted, '--', lw=1.0, label='negative boundary', color='#FF9900')

    # Plot support vectors
    plt.scatter(sv_x[:, 0], sv_x[:, 1], s=50, marker='o', c='white')

    # Add lambda values as annotations
    for s, (x1, x2) in zip(lamb, x):
        plt.annotate(f'λ={s[0]:.2f}', (x1 - 0.05, x2 + 0.2))

    # Encode plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
