#ifndef TIMESTATISTICS_H_
#define TIMESTATISTICS_H_

#include "timer.h"
#include "statistics.h"

class TimeStatistics
{
  public:
    explicit TimeStatistics(Statistics* stats=NULL, TimeType type=CPU_WALL_TIME);
    virtual
    ~TimeStatistics();

    int add(const std::string& label, const std::string& unit="ms", bool invert=false, double factor=1.0);
    int append(const std::string& label, const std::string& unit="ms", bool invert=false, double factor=1.0);
    void setFactor(int index, double factor);
    void setFactorAll(double factor);
    void start(int index);
    void stop(int index);
    const Statistics* statistics() const;

  private:
    Statistics* _statistics;
    std::vector<MTime> _mtimes;
    bool _delete_statistics;
    TimeType _timetype;
    std::vector<bool> _started;
};

inline void
TimeStatistics::setFactor(int index, double factor)
{
  _statistics->setFactor(index, factor);
}

inline void
TimeStatistics::setFactorAll(double factor)
{
  _statistics->setFactorAll(factor);
}

inline const Statistics* TimeStatistics::statistics() const{
  return _statistics;
}

#endif /* TIMESTATISTICS_H_ */
