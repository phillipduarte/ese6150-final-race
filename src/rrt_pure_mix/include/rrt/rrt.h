// RRT assignment

// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf

#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <array>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <tf2_ros/transform_broadcaster.h>

using namespace std;

// Struct defining the RRT_Node object in the RRT tree.
typedef struct RRT_Node {
    double x, y;
    double cost; // only used for RRT*
    int parent;  // index of parent node in the tree vector
    bool is_root = false;
} RRT_Node;


class RRT : public rclcpp::Node {
public:
    RRT();
    virtual ~RRT();
private:

    // Subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

    // Publishers
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr tree_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr waypoints_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;

    // Random generator
    std::mt19937 gen;
    std::uniform_real_distribution<> x_dist;
    std::uniform_real_distribution<> y_dist;
    std::uniform_real_distribution<> goal_bias_dist;  // for goal biasing

    // Occupancy grid (local, car-frame, row-major)
    std::vector<int8_t> occupancy_grid_;
    double grid_resolution_ = 0.1;  // meters per cell
    int grid_width_  = 200;         // cells (±10 m in x)
    int grid_height_ = 200;         // cells (±10 m in y)
    int inflation_radius_ = 3;      // cells to inflate obstacles by

    // Current car pose (updated each pose_callback)
    double car_x_   = 0.0;
    double car_y_   = 0.0;
    double car_yaw_ = 0.0;

    // Goal in car-local frame (updated each pose_callback)
    double goal_local_x_ = 0.0;
    double goal_local_y_ = 0.0;

    // Last successfully found path (kept for fallback visualization)
    std::vector<RRT_Node> last_path_;

    // Waypoints in map frame [x, y, yaw, v]
    std::vector<std::array<double, 4>> waypoints_;
    int pose_tick_ = 0;  // counts pose_callback calls (for low-rate publishing)

    // Tunable RRT parameters
    double max_expansion_dist_ = 0.5;
    double goal_threshold_     = 0.3;
    int    max_iterations_     = 500;
    double lookahead_dist_     = 2.0;
    double goal_bias_prob_     = 0.10;  // probability of sampling goal

    // RRT* parameters
    double search_radius_      = 1.0;
    double max_shortcut_dist_  = 1.0;  // max segment length allowed by shortcut_path
    bool   use_rrt_star_       = false;

    // Pure Pursuit parameters
    double pursuit_lookahead_     = 0.8;  // meters along path
    double wheelbase_             = 0.33; // F1TENTH wheelbase (meters)
    double max_speed_             = 1.5;  // m/s
    double waypoint_speed_scale_  = 0.8;  // multiplied against CSV speed in clear mode

    // callbacks
    void pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg);
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg);

    // RRT methods
    std::vector<double> sample();
    int nearest(std::vector<RRT_Node> &tree, std::vector<double> &sampled_point);
    RRT_Node steer(RRT_Node &nearest_node, std::vector<double> &sampled_point);
    bool check_collision(RRT_Node &nearest_node, RRT_Node &new_node);
    bool is_goal(RRT_Node &latest_added_node, double goal_x, double goal_y);
    std::vector<RRT_Node> find_path(std::vector<RRT_Node> &tree, RRT_Node &latest_added_node);

    // RRT* methods
    double cost(std::vector<RRT_Node> &tree, RRT_Node &node);
    double line_cost(RRT_Node &n1, RRT_Node &n2);
    std::vector<int> near(std::vector<RRT_Node> &tree, RRT_Node &node);

    // Visualization helpers
    void visualize_tree(std::vector<RRT_Node> &tree);
    void visualize_path(std::vector<RRT_Node> &path);
    void visualize_waypoints();
    void visualize_goal(double goal_local_x, double goal_local_y);

    // Path post-processing
    std::vector<RRT_Node> shortcut_path(std::vector<RRT_Node> &path);

    // Path execution
    void execute_path(std::vector<RRT_Node> &path);

    // Grid helpers
    bool world_to_grid(double wx, double wy, int &gx, int &gy);
    bool grid_in_bounds(int gx, int gy);
    bool grid_occupied(int gx, int gy);
    bool has_obstacle_in_corridor();

    // Waypoint helpers
    bool find_goal(double &goal_local_x, double &goal_local_y);
};