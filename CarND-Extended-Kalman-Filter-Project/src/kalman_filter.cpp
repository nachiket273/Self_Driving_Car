#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
   x_ = F_ * x_;
   MatrixXd ft = F_.transpose();
   P_ = F_*P_*ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
  VectorXd y = z - H_ * x_;
  UpdateEx(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];

  float sqt = sqrt(px*px + py*py);

  if(px == 0. && py == 0.) {
    return;
  }

  if( sqt < 0.0001) {
    // Add small esplilon to sqrt.
    sqt = 0.0001;
  }

  VectorXd hx(3);
  hx << sqt, atan2(py, px), (px* vx + py* vy)/sqt;

  VectorXd y = z - hx;

  // phi needs to in range between -pi and pi (atan range)
  while(y[1] > M_PI){
    y[1] -= 2 * M_PI;
  }
  while(y[1] < -M_PI){
    y[1] += 2 * M_PI;
  }

  UpdateEx(y);
}

void KalmanFilter::UpdateEx(const VectorXd &y) {
  MatrixXd ht = H_.transpose();
  MatrixXd S = H_ * P_ * ht + R_;
  MatrixXd si = S.inverse();
  MatrixXd K = P_ * ht * si;

  x_ = x_ + K * y;

  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K*H_)*P_;
}
