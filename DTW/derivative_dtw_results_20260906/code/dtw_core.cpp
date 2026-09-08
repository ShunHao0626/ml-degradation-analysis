#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
extern "C" void distances(const double* data,const long long* offsets,int n,double radius,int rms,double* out){
 const double inf=std::numeric_limits<double>::infinity();
 for(int a=0;a<n;a++){out[a*n+a]=0;
 for(int b=a+1;b<n;b++){
  int na=offsets[a+1]-offsets[a],nb=offsets[b+1]-offsets[b];
  const double* x=data+offsets[a]*2; const double* y=data+offsets[b]*2;
  std::vector<double> prev(nb+1,inf),curr(nb+1,inf);std::vector<int> lp(nb+1,0),lc(nb+1,0);prev[0]=0;
  // Radius in fractional sequence index; minimum half-cell width guarantees connectivity.
  double band=radius<0?2:std::max(radius,0.5/(na-1)+0.5/(nb-1)+1e-12);
  for(int i=0;i<na;i++){
   std::fill(curr.begin(),curr.end(),inf);std::fill(lc.begin(),lc.end(),0);
   int lo=std::max(0,(int)std::ceil(((double)i/(na-1)-band)*(nb-1)-1e-10));
   int hi=std::min(nb-1,(int)std::floor(((double)i/(na-1)+band)*(nb-1)+1e-10));
   for(int j=lo;j<=hi;j++){
    double best=prev[j];int len=lp[j];
    if(prev[j+1]<best){best=prev[j+1];len=lp[j+1];}
    if(curr[j]<best){best=curr[j];len=lc[j];}
    double dy=x[2*i]-y[2*j],dd=x[2*i+1]-y[2*j+1];
    curr[j+1]=best+dy*dy+dd*dd;lc[j+1]=len+1;
   }prev.swap(curr);lp.swap(lc);
  }
  out[a*n+b]=out[b*n+a]=std::sqrt(prev[nb]/(rms?lp[nb]:1));
 }
 }
}
