import{an as y,o as T,u as b,v as I,ao as w,k as C,j as t,G as n,O as R,B as o,aq as D,x as v,m as c,T as M,L as S,f as k}from"./vendor-10e6c6ad.js";import{i as u}from"./tiny-invariant-dd7d57d2.js";import{a as m}from"./createSvgIcon-54b26959.js";import{u as G,D as L,T as A,H as B,a as H}from"./DataGrid-7b9e973c.js";import{d as F}from"./dayjs.min-feca223a.js";import"./ContentCopy-38c812a0.js";const q=async({params:s})=>(u(s.projectId,"missing projectId"),m.getReports(s.projectId)),N=async({params:s})=>(u(s.projectId,"missing projectId"),m.reloadProject(s.projectId)),Q={crumb:(s,{pathname:i})=>({to:i,linkText:"Reports"})},U=()=>{const{projectId:s}=y(),i=T(),p=b(),x=I(),[f]=w(),[r,d]=C.useState(()=>{var e;return((e=f.get("tags"))==null?void 0:e.split(","))||[]});G("tags",r.join(","));const l=p.find(({id:e})=>e==="show-report-by-id"),j=l?[]:Array.from(new Set(i.flatMap(({tags:e})=>e))),h=i.filter(({tags:e})=>l?!1:r.length===0?!0:r.every(a=>e.includes(a))).map(e=>({id:e.id,"Report ID":e.id,tags:e.tags,timestamp:new Date(Date.parse(e.timestamp))})),g=[{field:"Report ID",flex:2,sortable:!1,renderCell:({row:e})=>t.jsx(o,{minHeight:73,display:"flex",alignItems:"center",children:t.jsx(A,{showText:e.id,copyText:e.id})})},{field:"Tags",flex:2,sortable:!1,renderCell:({row:e})=>t.jsx(o,{p:2,children:t.jsx(B,{onClick:a=>{r.includes(a)||d([...r,a])},tags:e.tags})})},{field:"timestamp",type:"dateTime",flex:1,renderCell({row:e}){return t.jsx(M,{variant:"body2",children:F(e.timestamp).locale("en-gb").format("llll")})}},{field:"Actions",flex:1,sortable:!1,renderCell({row:e}){return t.jsxs(t.Fragment,{children:[t.jsx(S,{component:k,to:`${e.id}`,children:t.jsx(c,{children:"View"})}),t.jsx(H,{downloadLink:`/api/projects/${s}/${e.id}/download`})]})}}];return l?t.jsx(n,{container:!0,children:t.jsx(n,{item:!0,xs:12,children:t.jsx(R,{})})}):t.jsxs(t.Fragment,{children:[t.jsx(o,{sx:{padding:2},children:t.jsxs(n,{container:!0,spacing:2,alignItems:"end",children:[t.jsx(n,{item:!0,xs:12,md:6,children:t.jsx(D,{multiple:!0,limitTags:2,value:r,onChange:(e,a)=>d(a),options:j,renderInput:e=>t.jsx(v,{...e,variant:"standard",label:"Filter by Tags"})})}),t.jsx(n,{item:!0,flexGrow:2,children:t.jsx(o,{display:"flex",justifyContent:"flex-end",children:t.jsx(c,{variant:"outlined",onClick:()=>x(null,{method:"post"}),color:"primary",children:"Refresh Reports"})})})]})}),t.jsx(L,{sx:{border:"none",[[".MuiDataGrid-cell",".MuiDataGrid-columnHeader"].flatMap(e=>[e+":focus",e+":focus-within"]).join(", ")]:{outline:"unset"}},initialState:{sorting:{sortModel:[{field:"timestamp",sort:"desc"}]}},disableRowSelectionOnClick:!0,disableColumnMenu:!0,getRowHeight:()=>"auto",density:"standard",columns:g,rows:h})]})};export{U as Component,N as action,Q as handle,q as loader};