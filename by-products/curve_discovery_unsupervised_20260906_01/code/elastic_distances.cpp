#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
extern "C" {
double raw_dtw(const double* a,const double* ta,int n,const double* b,const double* tb,int m,double beta){
    const double inf=std::numeric_limits<double>::infinity();
    std::vector<double> prev(m+1,inf),cur(m+1,inf);prev[0]=0;
    for(int i=1;i<=n;i++){
        cur[0]=inf;
        for(int j=1;j<=m;j++){
            double dy=a[i-1]-b[j-1],dt=ta[i-1]-tb[j-1];
            cur[j]=dy*dy+beta*dt*dt+std::min({prev[j],cur[j-1],prev[j-1]});
        }
        prev.swap(cur);
    }
    // Fixed n+m scaling; this is not minimization of average path cost.
    return std::sqrt(prev[m]/(n+m));
}
double msm_cost(double x,double a,double b,double c){
    if((a<=x && x<=b)||(b<=x && x<=a))return c;
    return c+std::min(std::abs(x-a),std::abs(x-b));
}
double raw_msm(const double* a,int n,const double* b,int m,double c){
    std::vector<double> prev(m),cur(m);prev[0]=std::abs(a[0]-b[0]);
    for(int j=1;j<m;j++)prev[j]=prev[j-1]+msm_cost(b[j],a[0],b[j-1],c);
    for(int i=1;i<n;i++){
        cur[0]=prev[0]+msm_cost(a[i],a[i-1],b[0],c);
        for(int j=1;j<m;j++)cur[j]=std::min({prev[j-1]+std::abs(a[i]-b[j]),
            prev[j]+msm_cost(a[i],a[i-1],b[j],c),cur[j-1]+msm_cost(b[j],a[i],b[j-1],c)});
        prev.swap(cur);
    }
    return prev[m-1];
}
}
