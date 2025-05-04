/* Consumer Demand Predictor - Professional Edition
 * 
 * Features:
 *  - Statistical utilities (mean, variance, normalization)
 *  - Abstract base classes for predictors
 *  - Multiple forecasting models (Linear Regression, ARIMA, Exponential Smoothing, etc.)
 *  - Model evaluation and hyperparameter optimization
 *  - Comprehensive unit testing
 *  - Clean separation of concerns and extensibility
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <limits>
#include <random>
#include <functional>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>

// Utility Functions

namespace stats {

inline double mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size());
}

inline double variance(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;
    double m = mean(data);
    double var = 0.0;
    for (double v : data) var += (v - m) * (v - m);
    return var / (data.size() - 1);
}

inline std::vector<double> minmax_scale(const std::vector<double>& data) {
    if (data.empty()) return data;
    double minVal = *std::min_element(data.begin(), data.end());
    double maxVal = *std::max_element(data.begin(), data.end());
    if (std::abs(maxVal - minVal) < 1e-12) return std::vector<double>(data.size(), 0.0);
    std::vector<double> scaled;
    scaled.reserve(data.size());
    for (double v : data) {
        scaled.push_back((v - minVal) / (maxVal - minVal));
    }
    return scaled;
}

inline std::vector<double> load_csv(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Could not open file: " + filename);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double value;
        if (ss >> value) data.push_back(value);
    }
    return data;
}

template <typename Predictor>
double rolling_origin_mse(const Predictor& predictor, const std::vector<double>& data, size_t min_train_size = 2) {
    if (data.size() <= min_train_size) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double mse = 0.0;
    size_t count = 0;
    for (size_t i = min_train_size; i < data.size(); ++i) {
        try {
            std::vector<double> train_data(data.begin(), data.begin() + i);
            double prediction = predictor(train_data);
            double error = prediction - data[i];
            mse += error * error;
            ++count;
        } catch (const std::exception&) {
            continue;
        }
    }
    return (count > 0) ? mse / static_cast<double>(count) : std::numeric_limits<double>::quiet_NaN();
}

inline void print_prediction(const std::string& model, double pred, bool success = true, const std::string& err = "") {
    if (success) {
        std::cout << "[RESULT] " << model << " predicted next consumer demand: " << pred << std::endl;
    } else {
        std::cerr << "[ERROR] " << model << " prediction failed: " << err << std::endl;
    }
}

inline void print_mse(const std::string& model, double mse) {
    if (std::isnan(mse)) {
        std::cout << "[INFO] " << model << " MSE: N/A (insufficient data)" << std::endl;
    } else {
        std::cout << "[METRIC] " << model << " Rolling-Origin MSE: " << mse << std::endl;
    }
} 

// Predictor Base Classes 

class IPredictor {
public:
    virtual ~IPredictor() = default;
    virtual double predict(const std::vector<double>& data) const = 0;
};

class IHyperparameterTunable {
public:
    virtual ~IHyperparameterTunable() = default;
    virtual void set_params(const std::map<std::string, double>& params) = 0;
};

// Model Implementations

class LinearRegressionPredictor : public IPredictor {
public:
    double predict(const std::vector<double>& data) const override {
        validateData(data);
        return calculateLinearRegression(data);
    }

    double calculateMSE(const std::vector<double>& data) const {
        if (data.size() < 2) throw std::invalid_argument("Insufficient data for MSE calculation.");
        double mse = 0.0;
        int count = 0;
        for (size_t i = 1; i < data.size(); ++i) {
            try {
                double predicted = predict(std::vector<double>(data.begin(), data.begin() + i));
                mse += std::pow(predicted - data[i], 2);
                ++count;
            } catch (const std::exception&) {
                continue;
            }
        }
        return (count > 0) ? mse / count : std::numeric_limits<double>::quiet_NaN();
    }

private:
    void validateData(const std::vector<double>& data) const {
        if (data.size() < 2) throw std::invalid_argument("Insufficient data to perform prediction.");
        if (std::all_of(data.begin(), data.end(), [&](double val) { return val == data.front(); }))
            throw std::invalid_argument("Constant demand data, unable to perform meaningful prediction.");
    }

    double calculateLinearRegression(const std::vector<double>& data) const {
        const int n = static_cast<int>(data.size());
        double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0;
        for (int i = 0; i < n; ++i) {
            sumX += i;
            sumY += data[i];
            sumXY += i * data[i];
            sumX2 += i * i;
        }
        double denom = n * sumX2 - sumX * sumX;
        if (std::abs(denom) < 1e-10)
            throw std::runtime_error("Calculation error: denominator is zero or near zero, unable to compute slope.");
        double slope = (n * sumXY - sumX * sumY) / denom;
        double intercept = (sumY - slope * sumX) / n;
        return slope * n + intercept;
    }
};

class ARIMAPredictor : public IPredictor, public IHyperparameterTunable {
public:
    ARIMAPredictor(int p = 1, int d = 0, int q = 0) : p_(p), d_(d), q_(q) {}

    double predict(const std::vector<double>& data) const override {
        if (data.size() <= static_cast<size_t>(p_ + d_))
            throw std::invalid_argument("Insufficient data for ARIMA prediction.");
        std::vector<double> diffed = performDifferencing(data, d_);
        std::vector<double> ar_coefs = estimateARCoefficients(diffed, p_);
        double ar_part = computeARPart(diffed, ar_coefs, p_);
        double ma_part = computeMAPart(diffed, ar_coefs, q_);
        double prediction = invertDifferencing(ar_part + ma_part, data.back(), d_);
        return prediction;
    }

    void set_params(const std::map<std::string, double>& params) override {
        if (params.count("p")) p_ = static_cast<int>(params.at("p"));
        if (params.count("d")) d_ = static_cast<int>(params.at("d"));
        if (params.count("q")) q_ = static_cast<int>(params.at("q"));
    }

private:
    int p_, d_, q_;

    std::vector<double> performDifferencing(const std::vector<double>& data, int d) const {
        std::vector<double> diffed = data;
        for (int diff = 0; diff < d; ++diff) {
            std::vector<double> temp;
            for (size_t i = 1; i < diffed.size(); ++i)
                temp.push_back(diffed[i] - diffed[i - 1]);
            diffed = std::move(temp);
        }
        return diffed;
    }

    double computeARPart(const std::vector<double>& diffed, const std::vector<double>& ar_coefs, int p) const {
        double ar_part = 0.0;
        for (int i = 0; i < p; ++i)
            ar_part += ar_coefs[i] * diffed[diffed.size() - 1 - i];
        return ar_part;
    }

    double computeMAPart(const std::vector<double>& diffed, const std::vector<double>& ar_coefs, int q) const {
        if (q <= 0 || diffed.size() <= static_cast<size_t>(p_ + q)) return 0.0;
        std::vector<double> residuals;
        residuals.reserve(diffed.size() - p_);
        for (size_t t = p_; t < diffed.size(); ++t) {
            double ar_pred = 0.0;
            for (int j = 0; j < p_; ++j)
                ar_pred += ar_coefs[j] * diffed[t - 1 - j];
            residuals.push_back(diffed[t] - ar_pred);
        }
        double ma_part = 0.0;
        int effective_q = std::min(q, static_cast<int>(residuals.size()));
        if (effective_q > 0) {
            for (int i = 0; i < effective_q; ++i)
                ma_part += residuals[residuals.size() - 1 - i];
            ma_part /= static_cast<double>(effective_q);
        }
        return ma_part;
    }

    double invertDifferencing(double prediction, double last, int d) const {
        for (int diff = 0; diff < d; ++diff) {
            prediction += last;
            last = prediction;
        }
        return prediction;
    }

    std::vector<double> estimateARCoefficients(const std::vector<double>& x, int p) const {
        std::vector<double> r(p + 1, 0.0);
        int N = x.size();
        double mean_x = stats::mean(x);
        for (int k = 0; k <= p; ++k) {
            for (int t = k; t < N; ++t)
                r[k] += (x[t] - mean_x) * (x[t - k] - mean_x);
            r[k] /= (N - k);
        }
        std::vector<std::vector<double>> R(p, std::vector<double>(p, 0.0));
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < p; ++j)
                R[i][j] = r[std::abs(i - j)];
        return gaussianElimination(R, std::vector<double>(r.begin() + 1, r.end()));
    }

    std::vector<double> gaussianElimination(std::vector<std::vector<double>> A, std::vector<double> b) const {
        const double EPS = 1e-12;
        int n = static_cast<int>(b.size());
        for (int i = 0; i < n; ++i) {
            int maxRow = i;
            double maxVal = std::abs(A[i][i]);
            for (int k = i + 1; k < n; ++k) {
                if (std::abs(A[k][i]) > maxVal) {
                    maxVal = std::abs(A[k][i]);
                    maxRow = k;
                }
            }
            if (maxVal < EPS) throw std::runtime_error("Singular matrix detected in Gaussian elimination.");
            if (maxRow != i) {
                std::swap(A[i], A[maxRow]);
                std::swap(b[i], b[maxRow]);
            }
            for (int k = i + 1; k < n; ++k) {
                double factor = A[k][i] / (A[i][i] + EPS);
                for (int j = i; j < n; ++j)
                    A[k][j] -= factor * A[i][j];
                b[k] -= factor * b[i];
            }
        }
        std::vector<double> x(n, 0.0);
        for (int i = n - 1; i >= 0; --i) {
            double sum = b[i];
            for (int j = i + 1; j < n; ++j)
                sum -= A[i][j] * x[j];
            if (std::abs(A[i][i]) < EPS)
                throw std::runtime_error("Zero diagonal element encountered during back substitution.");
            x[i] = sum / (A[i][i] + EPS);
        }
        return x;
    }
};

class ExponentialSmoothingPredictor : public IPredictor, public IHyperparameterTunable {
public:
    explicit ExponentialSmoothingPredictor(double alpha = 0.5) : alpha_(alpha) {
        if (alpha_ < 0.0 || alpha_ > 1.0)
            throw std::invalid_argument("Smoothing factor alpha must be between 0 and 1.");
    }

    double predict(const std::vector<double>& data) const override {
        if (data.empty())
            throw std::invalid_argument("The dataset provided for exponential smoothing is empty.");
        double smoothedValue = data[0];
        for (size_t i = 1; i < data.size(); ++i)
            smoothedValue = alpha_ * data[i] + (1.0 - alpha_) * smoothedValue;
        return smoothedValue;
    }

    void set_params(const std::map<std::string, double>& params) override {
        if (params.count("alpha")) {
            double a = params.at("alpha");
            if (a < 0.0 || a > 1.0)
                throw std::invalid_argument("Smoothing factor alpha must be between 0 and 1.");
            alpha_ = a;
        }
    }

private:
    double alpha_;
};

class DecisionTreeStumpRegressor : public IPredictor {
public:
    double predict(const std::vector<double>& data) const override {
        if (data.size() < 2) throw std::invalid_argument("Insufficient data for decision tree stump.");
        double best_mse = std::numeric_limits<double>::max();
        double best_pred = stats::mean(data);
        for (size_t split = 1; split < data.size(); ++split) {
            std::vector<double> left(data.begin(), data.begin() + split);
            std::vector<double> right(data.begin() + split, data.end());
            double left_mean = stats::mean(left);
            double right_mean = stats::mean(right);
            double mse = 0.0;
            for (double v : left) mse += (v - left_mean) * (v - left_mean);
            for (double v : right) mse += (v - right_mean) * (v - right_mean);
            if (mse < best_mse) {
                best_mse = mse;
                best_pred = (right.empty()) ? left_mean : right_mean;
            }
        }
        return best_pred;
    }
};

class KNNRegressor : public IPredictor, public IHyperparameterTunable {
public:
    explicit KNNRegressor(int k = 2) : k_(k) {}

    double predict(const std::vector<double>& data) const override {
        if (data.size() < static_cast<size_t>(k_))
            throw std::invalid_argument("Insufficient data for KNN.");
        double sum = 0.0;
        for (size_t i = data.size() - k_; i < data.size(); ++i)
            sum += data[i];
        return sum / k_;
    }

    void set_params(const std::map<std::string, double>& params) override {
        if (params.count("k")) {
            int k = static_cast<int>(params.at("k"));
            if (k < 1) throw std::invalid_argument("k must be >= 1");
            k_ = k;
        }
    }

private:
    int k_;
};

class AR2Predictor : public IPredictor {
public:
    double predict(const std::vector<double>& data) const override {
        if (data.size() < 3) throw std::invalid_argument("Insufficient data for AR(2) prediction.");
        std::vector<double> r(3, 0.0);
        double mean_data = stats::mean(data);
        for (int k = 0; k <= 2; ++k) {
            for (size_t t = k; t < data.size(); ++t)
                r[k] += (data[t] - mean_data) * (data[t - k] - mean_data);
            r[k] /= (data.size() - k);
        }
        std::vector<std::vector<double>> R = {{r[0], r[1]}, {r[1], r[0]}};
        std::vector<double> phi = gaussianElimination(R, {r[1], r[2]});
        return phi[0] * data[data.size() - 1] + phi[1] * data[data.size() - 2];
    }
private:
    std::vector<double> gaussianElimination(std::vector<std::vector<double>> A, std::vector<double> b) const {
        const double EPS = 1e-12;
        int n = static_cast<int>(b.size());
        for (int i = 0; i < n; ++i) {
            int maxRow = i;
            double maxVal = std::abs(A[i][i]);
            for (int k = i + 1; k < n; ++k) {
                if (std::abs(A[k][i]) > maxVal) {
                    maxVal = std::abs(A[k][i]);
                    maxRow = k;
                }
            }
            if (maxVal < EPS) throw std::runtime_error("Singular matrix detected in Gaussian elimination.");
            if (maxRow != i) {
                std::swap(A[i], A[maxRow]);
                std::swap(b[i], b[maxRow]);
            }
            for (int k = i + 1; k < n; ++k) {
                double factor = A[k][i] / (A[i][i] + EPS);
                for (int j = i; j < n; ++j)
                    A[k][j] -= factor * A[i][j];
                b[k] -= factor * b[i];
            }
        }
        std::vector<double> x(n, 0.0);
        for (int i = n - 1; i >= 0; --i) {
            double sum = b[i];
            for (int j = i + 1; j < n; ++j)
                sum -= A[i][j] * x[j];
            if (std::abs(A[i][i]) < EPS)
                throw std::runtime_error("Zero diagonal element encountered during back substitution.");
            x[i] = sum / (A[i][i] + EPS);
        }
        return x;
    }
};

class HoltWintersPredictor : public IPredictor, public IHyperparameterTunable {
public:
    HoltWintersPredictor(double alpha = 0.5, double beta = 0.3) : alpha_(alpha), beta_(beta) {}

    double predict(const std::vector<double>& data) const override {
        if (data.size() < 2) throw std::invalid_argument("Insufficient data for Holt-Winters prediction.");
        double level = data[0];
        double trend = data[1] - data[0];
        for (size_t i = 1; i < data.size(); ++i) {
            double prevLevel = level;
            level = alpha_ * data[i] + (1 - alpha_) * (level + trend);
            trend = beta_ * (level - prevLevel) + (1 - beta_) * trend;
        }
        return level + trend;
    }

    void set_params(const std::map<std::string, double>& params) override {
        if (params.count("alpha")) {
            double a = params.at("alpha");
            if (a < 0.0 || a > 1.0) throw std::invalid_argument("alpha must be in [0,1]");
            alpha_ = a;
        }
        if (params.count("beta")) {
            double b = params.at("beta");
            if (b < 0.0 || b > 1.0) throw std::invalid_argument("beta must be in [0,1]");
            beta_ = b;
        }
    }

private:
    double alpha_, beta_;
};

class ProphetPredictor : public IPredictor, public IHyperparameterTunable {
public:
    ProphetPredictor(double growth = 0.1, double changepoint_prior_scale = 0.05)
        : growth_(growth), changepoint_prior_scale_(changepoint_prior_scale) {}

    double predict(const std::vector<double>& data) const override {
        if (data.size() < 2)
            throw std::invalid_argument("Insufficient data for Prophet prediction.");
        double trend = growth_ * (data.size() - 1);
        double base = data.front();
        double prediction = base + trend;
        double cumulativeTrendAdjustment = 0.0;
        for (size_t i = 1; i < data.size(); ++i) {
            double localTrend = growth_ * i;
            cumulativeTrendAdjustment += localTrend * (data[i] - data[i - 1]);
        }
        prediction += cumulativeTrendAdjustment / data.size();
        for (size_t i = 1; i < data.size(); ++i) {
            if (std::abs(data[i] - data[i - 1]) > changepoint_prior_scale_)
                prediction += (data[i] - data[i - 1]);
        }
        return prediction;
    }

    void set_params(const std::map<std::string, double>& params) override {
        if (params.count("growth")) growth_ = params.at("growth");
        if (params.count("changepoint_prior_scale")) changepoint_prior_scale_ = params.at("changepoint_prior_scale");
    }

private:
    double growth_;
    double changepoint_prior_scale_;
};

class RandomForestStumpRegressor : public IPredictor, public IHyperparameterTunable {
public:
    explicit RandomForestStumpRegressor(int n_estimators = 10) : n_estimators_(n_estimators) {
        if (n_estimators_ <= 0)
            throw std::invalid_argument("Number of estimators must be positive.");
    }

    double predict(const std::vector<double>& data) const override {
        if (data.size() < 2)
            throw std::invalid_argument("Insufficient data for Random Forest prediction.");
        double totalPrediction = 0.0;
        for (int i = 0; i < n_estimators_; ++i) {
            size_t split = 1 + (i % (data.size() - 1));
            double leftMean = std::accumulate(data.begin(), data.begin() + split, 0.0) / split;
            double rightMean = std::accumulate(data.begin() + split, data.end(), 0.0) / (data.size() - split);
            totalPrediction += (leftMean + rightMean) / 2.0;
        }
        return totalPrediction / n_estimators_;
    }

    void set_params(const std::map<std::string, double>& params) override {
        if (params.count("n_estimators")) {
            int n = static_cast<int>(params.at("n_estimators"));
            if (n <= 0) throw std::invalid_argument("n_estimators must be positive.");
            n_estimators_ = n;
        }
    }

private:
    int n_estimators_;
};

// Hybrid Ensemble Predictor

class HybridEnsemblePredictor : public IPredictor {
public:
    struct ModelWeight {
        std::string name;
        double weight;
    };

    HybridEnsemblePredictor(
        double linear_weight = 0.4,
        double arima_weight = 0.3,
        double exp_weight = 0.2,
        double knn_weight = 0.1
    )
        : weights_{
            {"LinearRegression", linear_weight},
            {"ARIMA", arima_weight},
            {"ExponentialSmoothing", exp_weight},
            {"KNN", knn_weight}
        }
    {
        double total = 0.0;
        for (const auto& w : weights_) total += w.weight;
        if (std::abs(total) < 1e-8) throw std::invalid_argument("Sum of model weights must be positive.");
    }

    double predict(const std::vector<double>& data) const override {
        if (data.size() < 3)
            throw std::invalid_argument("HybridEnsemblePredictor requires at least 3 data points.");
        std::vector<std::unique_ptr<IPredictor>> models;
        models.emplace_back(std::make_unique<LinearRegressionPredictor>());
        models.emplace_back(std::make_unique<ARIMAPredictor>(1, 0, 0));
        models.emplace_back(std::make_unique<ExponentialSmoothingPredictor>(0.5));
        models.emplace_back(std::make_unique<KNNRegressor>(2));
        std::vector<double> predictions;
        predictions.reserve(models.size());
        for (const auto& model : models) {
            try {
                predictions.push_back(model->predict(data));
            } catch (...) {
                predictions.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }
        double weighted_sum = 0.0;
        double total_weight = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (!std::isnan(predictions[i])) {
                weighted_sum += predictions[i] * weights_[i].weight;
                total_weight += weights_[i].weight;
            }
        }
        if (total_weight == 0.0)
            throw std::runtime_error("All base predictors failed in HybridEnsemblePredictor.");
        return weighted_sum / total_weight;
    }

    const std::vector<ModelWeight>& get_weights() const { return weights_; }

private:
    std::vector<ModelWeight> weights_;
};

// Hyperparameter Tuning

int grid_search_knn_k(const std::vector<double>& data, const std::vector<int>& k_values) {
    double best_mse = std::numeric_limits<double>::max();
    int best_k = k_values.front();
    for (int k : k_values) {
        KNNRegressor knn(k);
        double mse = stats::rolling_origin_mse([&](const std::vector<double>& d) { return knn.predict(d); }, data, k);
        if (!std::isnan(mse) && mse < best_mse) {
            best_mse = mse;
            best_k = k;
        }
    }
    return best_k;
}

std::tuple<int, int, int> grid_search_arima(const std::vector<double>& data, const std::vector<int>& p_values, const std::vector<int>& d_values, const std::vector<int>& q_values) {
    double best_mse = std::numeric_limits<double>::max();
    int best_p = 1, best_d = 0, best_q = 0;
    for (int p : p_values) {
        for (int d : d_values) {
            for (int q : q_values) {
                try {
                    ARIMAPredictor arima(p, d, q);
                    double mse = stats::rolling_origin_mse([&](const std::vector<double>& dvec) { return arima.predict(dvec); }, data, std::max<size_t>(p + d, 2));
                    if (!std::isnan(mse) && mse < best_mse) {
                        best_mse = mse;
                        best_p = p; best_d = d; best_q = q;
                    }
                } catch (const std::exception&) {
                    continue;
                }
            }
        }
    }
    return std::make_tuple(best_p, best_d, best_q);
}

// Main Analysis Function 
void analyzeConsumerData(const std::vector<double>& consumerData) {
    using namespace stats;
    if (consumerData.empty()) {
        std::cerr << "Error: No data provided for analysis." << std::endl;
        return;
    }

    // Descriptive Analysis
    double m = mean(consumerData);
    double stdev = std::sqrt(variance(consumerData));
    std::cout << "Descriptive Analysis:\nMean: " << m << "\nStandard Deviation: " << stdev << std::endl;

    // Exploratory Data Analysis (EDA)
    std::cout << "Exploratory Data Analysis:\nData Points: ";
    for (const auto& value : consumerData) std::cout << value << " ";
    std::cout << std::endl;

    // Linear Regression prediction.
    LinearRegressionPredictor linearPredictor;
    double linPred = 0.0;
    bool linSuccess = true;
    std::string linErr;
    try {
        linPred = linearPredictor.predict(consumerData);
    } catch (const std::exception& e) {
        linSuccess = false;
        linErr = e.what();
    }
    print_prediction("Linear Regression", linPred, linSuccess, linErr);

    // ARIMA (grid search)
    auto best_arima = grid_search_arima(consumerData, {1,2}, {0,1}, {0,1});
    int best_p = std::get<0>(best_arima), best_d = std::get<1>(best_arima), best_q = std::get<2>(best_arima);
    ARIMAPredictor arima(best_p, best_d, best_q);
    double arimaPred = 0.0;
    bool arimaSuccess = true;
    std::string arimaErr;
    try {
        arimaPred = arima.predict(consumerData);
    } catch (const std::exception& e) {
        arimaSuccess = false;
        arimaErr = e.what();
    }
    print_prediction("ARIMA (p=" + std::to_string(best_p) + ",d=" + std::to_string(best_d) + ",q=" + std::to_string(best_q) + ")", arimaPred, arimaSuccess, arimaErr);

    // Exponential Smoothing (alpha grid search)
    double best_alpha = 0.5, best_exp_mse = std::numeric_limits<double>::max();
    for (double alpha : {0.2, 0.5, 0.8}) {
        ExponentialSmoothingPredictor expSmooth(alpha);
        double mse = rolling_origin_mse([&](const std::vector<double>& d) { return expSmooth.predict(d); }, consumerData, 2);
        if (!std::isnan(mse) && mse < best_exp_mse) {
            best_exp_mse = mse;
            best_alpha = alpha;
        }
    }
    ExponentialSmoothingPredictor expSmooth(best_alpha);
    double expPred = 0.0;
    bool expSuccess = true;
    std::string expErr;
    try {
        expPred = expSmooth.predict(consumerData);
    } catch (const std::exception& e) {
        expSuccess = false;
        expErr = e.what();
    }
    print_prediction("Exponential Smoothing (alpha=" + std::to_string(best_alpha) + ")", expPred, expSuccess, expErr);

    // Decision Tree Stump prediction.
    DecisionTreeStumpRegressor dtStump;
    double dtPred = 0.0;
    bool dtSuccess = true;
    std::string dtErr;
    try {
        dtPred = dtStump.predict(consumerData);
    } catch (const std::exception& e) {
        dtSuccess = false;
        dtErr = e.what();
    }
    print_prediction("Decision Tree Stump", dtPred, dtSuccess, dtErr);

    // KNN prediction (with grid search for best k).
    int best_k = grid_search_knn_k(consumerData, {2,3,4});
    KNNRegressor knn(best_k);
    double knnPred = 0.0;
    bool knnSuccess = true;
    std::string knnErr;
    try {
        knnPred = knn.predict(consumerData);
    } catch (const std::exception& e) {
        knnSuccess = false;
        knnErr = e.what();
    }
    print_prediction("KNN (k=" + std::to_string(best_k) + ")", knnPred, knnSuccess, knnErr);

    // Print MSE for Linear Regression.
    try {
        double mse = linearPredictor.calculateMSE(consumerData);
        print_mse("Linear Regression", mse);
    } catch (const std::exception& e) {
        std::cerr << "Error in model evaluation: " << e.what() << std::endl;
    }

    // AR(2) prediction and MSE.
    {
        AR2Predictor ar2;
        double ar2Pred = 0.0;
        bool ar2Success = true;
        std::string ar2Err;
        try {
            ar2Pred = ar2.predict(consumerData);
        } catch (const std::exception& e) {
            ar2Success = false;
            ar2Err = e.what();
        }
        print_prediction("AR(2)", ar2Pred, ar2Success, ar2Err);

        double ar2_mse = rolling_origin_mse(
            [&](const std::vector<double>& d) { return ar2.predict(d); },
            consumerData, 3
        );
        print_mse("AR(2)", ar2_mse);
    }

    // Holt-Winters Prediction & Evaluation
    {
        HoltWintersPredictor hw;
        double best_hw_alpha = 0.5, best_hw_beta = 0.3, best_hw_mse = std::numeric_limits<double>::max();
        for (double alpha : {0.2, 0.5, 0.8}) {
            for (double beta : {0.1, 0.3, 0.5}) {
                HoltWintersPredictor hw(alpha, beta);
                double mse = rolling_origin_mse(
                    [&](const std::vector<double>& d) { return hw.predict(d); },
                    consumerData, 2
                );
                if (!std::isnan(mse) && mse < best_hw_mse) {
                    best_hw_mse = mse;
                    best_hw_alpha = alpha;
                    best_hw_beta = beta;
                }
            }
        }
        HoltWintersPredictor hw(best_hw_alpha, best_hw_beta);
        double hwPred = 0.0;
        bool hwSuccess = true;
        std::string hwErr;
        try {
            hwPred = hw.predict(consumerData);
        } catch (const std::exception& e) {
            hwSuccess = false;
            hwErr = e.what();
        }
        print_prediction("Holt-Winters (alpha=" + std::to_string(best_hw_alpha) + ", beta=" + std::to_string(best_hw_beta) + ")", hwPred, hwSuccess, hwErr);

        double hw_mse = rolling_origin_mse(
            [&](const std::vector<double>& d) { return hw.predict(d); },
            consumerData, 2
        );
        print_mse("Holt-Winters", hw_mse);
    }

    // Decision Tree Stump and Random Forest Stump predictions and MSE.
    {
        DecisionTreeStumpRegressor dt;
        double dtPred = 0.0;
        bool dtSuccess = true;
        std::string dtErr;
        try {
            dtPred = dt.predict(consumerData);
        } catch (const std::exception& e) {
            dtSuccess = false;
            dtErr = e.what();
        }
        print_prediction("Decision Tree Stump", dtPred, dtSuccess, dtErr);

        RandomForestStumpRegressor rf(10);
        double rf_mse = rolling_origin_mse(
            [&](const std::vector<double>& d) { return rf.predict(d); },
            consumerData, 2
        );
        print_mse("Random Forest Stump", rf_mse);
    }

    // Data Normalization (Min-Max Scaling)
    {
        std::vector<double> scaledData = minmax_scale(consumerData);
        std::cout << "[PREPROCESS] Data Normalization (Min-Max Scaling):\nScaled consumer data: ";
        for (double v : scaledData) std::cout << v << " ";
        std::cout << std::endl;
    }

    // Hybrid Ensemble prediction.
    {
        HybridEnsemblePredictor hybridModel;
        double hybridPred = 0.0;
        bool hybridSuccess = true;
        std::string hybridErr;
        try {
            hybridPred = hybridModel.predict(consumerData);
        } catch (const std::exception& e) {
            hybridSuccess = false;
            hybridErr = e.what();
        }
        print_prediction("Hybrid Ensemble", hybridPred, hybridSuccess, hybridErr);
    }
}

// Unit Tests 

void run_unit_tests() {
    using namespace stats;
    std::cout << "\n[UNIT TESTS] Running unit tests...\n";
    // Test mean/variance
    assert(mean({}) == 0.0);
    assert(std::abs(mean({1,2,3}) - 2.0) < 1e-8);
    assert(std::abs(variance({1,2,3}) - 1.0) < 1e-8);

    // Test Linear Regression
    LinearRegressionPredictor lr;
    try { lr.predict({}); assert(false && "Should throw on empty data"); } catch (...) {}
    try { lr.predict({1,1,1}); assert(false && "Should throw on constant data"); } catch (...) {}
    assert(std::abs(lr.predict({1,2,3,4,5}) - 6.0) < 1e-6);

    // Test ARIMA
    ARIMAPredictor arima(1,0,0);
    try { arima.predict({1}); assert(false && "Should throw on too little data"); } catch (...) {}
    double arima_pred = arima.predict({1,2,3,4,5});
    assert(std::isfinite(arima_pred));

    // Test Exponential Smoothing
    ExponentialSmoothingPredictor exp(0.5);
    try { exp.predict({}); assert(false && "Should throw on empty data"); } catch (...) {}
    assert(std::abs(exp.predict({1,2,3,4,5}) - 4.0625) < 1e-4);

    // Test KNN
    KNNRegressor knn(2);
    try { knn.predict({1}); assert(false && "Should throw on too little data"); } catch (...) {}
    assert(std::abs(knn.predict({1,2,3,4,5}) - 4.5) < 1e-8);

    // Test Decision Tree Stump
    DecisionTreeStumpRegressor dt;
    try { dt.predict({1}); assert(false && "Should throw on too little data"); } catch (...) {}
    assert(std::isfinite(dt.predict({1,2,3,4,5})));

    // Test AR2
    AR2Predictor ar2;
    try { ar2.predict({1,2}); assert(false && "Should throw on too little data"); } catch (...) {}
    assert(std::isfinite(ar2.predict({1,2,3,4,5})));

    // Test Holt-Winters
    HoltWintersPredictor hw;
    try { hw.predict({1}); assert(false && "Should throw on too little data"); } catch (...) {}
    assert(std::isfinite(hw.predict({1,2,3,4,5})));

    // Test Prophet
    ProphetPredictor prophet;
    try { prophet.predict({1}); assert(false && "Should throw on too little data"); } catch (...) {}
    assert(std::isfinite(prophet.predict({1,2,3,4,5})));

    // Test HybridEnsemble
    HybridEnsemblePredictor hybrid;
    try { hybrid.predict({1,2}); assert(false && "Should throw on too little data"); } catch (...) {}
    assert(std::isfinite(hybrid.predict({1,2,3,4,5})));

    // Test minmax_scale
    auto scaled = minmax_scale({1,2,3});
    assert(std::abs(scaled.front() - 0.0) < 1e-8 && std::abs(scaled.back() - 1.0) < 1e-8);

    std::cout << "[UNIT TESTS] All tests passed!\n";
}

//Main Function  

int main(int argc, char* argv[]) {
    try {
        std::vector<double> consumerData;
        if (argc > 1) {
            consumerData = stats::load_csv(argv[1]);
            std::cout << "[INFO] Loaded " << consumerData.size() << " data points from file: " << argv[1] << std::endl;
        } else {
            // Use static data as fallback
            consumerData = {200, 210, 220, 230, 240};
            std::cout << "[INFO] Using static sample data.\n";
        }
        analyzeConsumerData(consumerData);
        run_unit_tests();
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
    }
    return 0;
}
