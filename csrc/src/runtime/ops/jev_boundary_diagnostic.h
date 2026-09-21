#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include "utilities/tensor.h"
namespace dsl::jev_diagnostic {
using Json=nlohmann::json;
inline int layer_from_name(const std::string&name,int fallback){const auto begin=name.find("blocks[");if(begin==std::string::npos)return fallback;const auto end=name.find(']',begin);return std::stoi(name.substr(begin+7,end-begin-7));}
inline bool enabled(){return std::getenv("JEV_GEMMA_BOUNDARY_DIAGNOSTIC")!=nullptr;}
inline std::vector<double> read(const Tensor&t,cudaStream_t stream){
 std::vector<double> out(t.nelem());
 if(t.DType==ETensorDType::BF16){std::vector<nv_bfloat16> h(t.nelem());CUDA_CHECK(cudaMemcpyAsync(h.data(),t.Data,t.bytes(),cudaMemcpyDeviceToHost,stream));CUDA_CHECK(cudaStreamSynchronize(stream));for(size_t i=0;i<h.size();++i)out[i]=__bfloat162float(h[i]);}
 else if(t.DType==ETensorDType::FP32){std::vector<float> h(t.nelem());CUDA_CHECK(cudaMemcpyAsync(h.data(),t.Data,t.bytes(),cudaMemcpyDeviceToHost,stream));CUDA_CHECK(cudaStreamSynchronize(stream));std::copy(h.begin(),h.end(),out.begin());}
 else throw std::runtime_error("diagnostic unexpected tensor dtype");return out;
}
inline Json stats(const std::vector<double>&v){double sum=0,max=0;size_t nonfinite=0;for(double x:v){if(!std::isfinite(x))++nonfinite;else{sum+=x*x;max=std::max(max,std::abs(x));}}return Json{{"l2",std::sqrt(sum)},{"max_abs",max},{"nonfinite",nonfinite},{"numel",v.size()}};}
inline Json token_support(const std::vector<double>&v){
 if(v.size()%256)return Json{{"shape_not_256_tokens",true}};
 const size_t width=v.size()/256;Json nz=Json::array(),l2=Json::array();double tailmax=0;size_t tailnz=0;Json samples=Json::array();
 for(size_t t=0;t<256;++t){size_t count=0;double square=0;for(size_t c=0;c<width;++c){double x=v[t*width+c];count+=x!=0;square+=x*x;if(t>=222&&x!=0){++tailnz;tailmax=std::max(tailmax,std::abs(x));if(samples.size()<16)samples.push_back(Json{{"token",t},{"column",c},{"value",x}});}}nz.push_back(count);l2.push_back(std::sqrt(square));}
 return Json{{"tokens",256},{"width",width},{"token_nonzero",nz},{"token_l2",l2},{"tail_nonzero",tailnz},{"tail_max_abs",tailmax},{"tail_first16_nonzero",samples}};
}
inline std::uint64_t fingerprint(const std::vector<double>&v){std::uint64_t h=1469598103934665603ull;const auto*p=reinterpret_cast<const unsigned char*>(v.data());for(size_t i=0;i<v.size()*sizeof(double);++i){h^=p[i];h*=1099511628211ull;}return h;}
inline void emit(Json row,int device){const char*dir=std::getenv("JEV_GEMMA_BOUNDARY_DIAGNOSTIC");if(!dir)return;std::ofstream f(std::string(dir)+"/boundary-gpu"+std::to_string(device)+".jsonl",std::ios::app);f<<row.dump()<<'\n';if(!f)throw std::runtime_error("diagnostic write failed");}
struct ScalarSnapshot{std::vector<double>x;double scale=0;long columns=0;};
inline ScalarSnapshot scalar_before(const Tensor&x,const Tensor&scale,cudaStream_t stream){if(!enabled())return {};return {read(x,stream),read(scale,stream).at(0),x.Sizes[x.Rank-1]};}
inline void scalar_after(const ScalarSnapshot&s,const Tensor&after,int layer,int device,cudaStream_t stream,bool backward,bool replay){
 if(s.x.empty())return;auto y=read(after,stream);const long prefix=std::min<long>(s.x.size(),222*s.columns);std::vector<double>xp(s.x.begin(),s.x.begin()+prefix),yp(y.begin(),y.begin()+prefix);double err=0,ref=0;for(size_t i=0;i<s.x.size();++i){double e=s.x[i]*s.scale;err+=(y[i]-e)*(y[i]-e);ref+=e*e;}emit(Json{{"kind",backward?"scalar_backward":"scalar_forward"},{"layer",layer},{"replay",replay},{"scalar",s.scale},{"before",stats(s.x)},{"after",stats(y)},{"before_prefix222",stats(xp)},{"after_prefix222",stats(yp)},{"relative_formula_error",std::sqrt(err/std::max(ref,1e-300))}},device);
}
struct Snapshot{std::vector<double>x,w,r,dy,dr;int rows=0,C=0;double eps=0;Json row;};
inline Snapshot rms_before(const Tensor&x,const Tensor&w,const Tensor&r,const Tensor*dy,const Tensor*dr,int layer,const std::string&key,const std::string&weight,int device,cudaStream_t stream,double eps,bool replay,const std::vector<double>*forward_x=nullptr){
 Snapshot s;if(!enabled()||layer<26)return s;s.x=forward_x?*forward_x:read(x,stream);s.w=read(w,stream);s.r=read(r,stream);s.C=s.w.size();s.rows=s.x.size()/s.C;s.eps=eps;if(dy)s.dy=read(*dy,stream);if(dr)s.dr=read(*dr,stream);
 double maxrel=0,err=0,ref=0;size_t bad=0;for(int row=0;row<s.rows;++row){double square=0;for(int c=0;c<s.C;++c){double value=s.x[row*s.C+c];square+=value*value;}double expected=1/std::sqrt(square/s.C+eps);double rel=std::abs(s.r[row]-expected)/std::max(expected,1e-300);maxrel=std::max(maxrel,rel);bad+=rel>1e-3;err+=(s.r[row]-expected)*(s.r[row]-expected);ref+=expected*expected;}
 s.row=Json{{"kind",dy?"rms_backward":"rms_forward"},{"layer",layer},{"key",key},{"weight",weight},{"replay",replay},{"rows",s.rows},{"C",s.C},{"eps",eps},{"x",stats(s.x)},{"x_hash",fingerprint(s.x)},{"rstd_hash",fingerprint(s.r)},{"rstd",stats(s.r)},{"rstd_max_relative_error",maxrel},{"rstd_relative_l2_error",std::sqrt(err/std::max(ref,1e-300))},{"rstd_rows_over_1e_3",bad}};
 if(dy){s.row["dy"]=stats(s.dy);s.row["dy_support"]=token_support(s.dy);if(dr)s.row["residual_adjoint_support"]=token_support(s.dr);}else emit(s.row,device);return s;
}
inline void rms_after(Snapshot&s,const Tensor&dx,int device,cudaStream_t stream){
 if(s.x.empty())return;auto actual=read(dx,stream);double err_saved=0,ref_saved=0,err_recomputed=0,ref_recomputed=0;
 for(int row=0;row<s.rows;++row){double square=0,dot=0;for(int c=0;c<s.C;++c){int i=row*s.C+c;square+=s.x[i]*s.x[i];dot+=s.dy[i]*s.w[c]*s.x[i];}double recomputed=1/std::sqrt(square/s.C+s.eps);for(int c=0;c<s.C;++c){int i=row*s.C+c;double residual=s.dr.empty()?0:s.dr[i];auto formula=[&](double r){return residual+r*(s.dy[i]*s.w[c]-s.x[i]*r*r*dot/s.C);};double a=formula(s.r[row]),b=formula(recomputed);err_saved+=(actual[i]-a)*(actual[i]-a);ref_saved+=a*a;err_recomputed+=(actual[i]-b)*(actual[i]-b);ref_recomputed+=b*b;}}
 s.row["dx"]=stats(actual);s.row["dx_support"]=token_support(actual);s.row["dx_relative_formula_error_saved_rstd"]=std::sqrt(err_saved/std::max(ref_saved,1e-300));s.row["dx_relative_formula_error_recomputed_rstd"]=std::sqrt(err_recomputed/std::max(ref_recomputed,1e-300));emit(s.row,device);
}
}
