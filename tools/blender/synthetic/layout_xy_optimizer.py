"""Fast deterministic XY search for the fixed PWA capture guides.

This module does not render trial images and does not move the guides.
Camera orientation, height, lens and tile rotations stay unchanged.
Only camera XY and the independent dora group XY may be translated.
"""
from __future__ import annotations
import math
from mathutils import Vector

def _intersection_area(box, rect):
    l,t,r,b=box
    x,y,w,h=rect
    return max(0.0,min(r,x+w)-max(l,x))*max(0.0,min(b,y+h)-max(t,y))

def _center(tiles, boxes, region):
    bs=[boxes[i] for i,t in enumerate(tiles) if t["region"]==region]
    if not bs: return None
    return ((min(b[0] for b in bs)+max(b[2] for b in bs))/2,
            (min(b[1] for b in bs)+max(b[3] for b in bs))/2)

class XYGuideOptimizer:
    def __init__(self, ctx, tiles, crops, g, max_foreign_ratio=0.10):
        self.tiles=tiles
        self.crops=crops
        self.g=g
        self.max_foreign_ratio=max_foreign_ratio
        cam=ctx["camera"]
        sc=ctx["scene"]
        self.cam=cam
        self.origin=cam.location.copy()
        self.camera_rotation=cam.rotation_euler.to_quaternion()
        self.inverse_rotation=self.camera_rotation.inverted()
        self.fx=sc.render.resolution_x*cam.data.lens/cam.data.sensor_width
        self.cx=sc.render.resolution_x/2
        self.cy=sc.render.resolution_y/2
        self.local_corners=[]
        self.groups=[]
        self.polys=[]
        for t in tiles:
            pts=[]
            for obj in t["children"]:
                if obj.type != "MESH": continue
                for p in obj.bound_box:
                    w=obj.matrix_world@Vector(p)
                    q=self.inverse_rotation@(w-self.origin)
                    pts.append((q.x,q.y,q.z))
            self.local_corners.append(pts)
            self.groups.append(t["region"])
            self.polys.append(g.obb_corners(t["x"],t["y"],math.radians(t["yaw_deg"])))
        self.right_axis=self.inverse_rotation@Vector((1,0,0))
        self.up_axis=self.inverse_rotation@Vector((0,1,0))
        # Black table-frame strips are routinely visible in real meld crops.
        # Reward their visibility softly, never at the cost of losing a tile.
        near_y=-g.TABLE_Y/2-.016
        right_x=g.TABLE_X/2+.016
        self.frame_samples=[]
        for kind,points in (
            ("near",[(x,near_y,.020) for x in (.22,.29,.36)]),
            ("right",[(right_x,y,.020) for y in (-.34,-.28,-.22)]),
        ):
            for point in points:
                v=self.inverse_rotation@(Vector(point)-self.origin)
                self.frame_samples.append((kind,(v.x,v.y,v.z)))

    def _centers_for_dora(self):
        return [(p[0],p[1]) for t,poly in zip(self.tiles,self.polys)
                if t["region"]=="dora_indicators" for p in poly]

    def boxes(self, cx,cy, dx,dy):
        boxes=[]
        for region,pts in zip(self.groups,self.local_corners):
            tx=(dx if region=="dora_indicators" else 0.0)-cx
            ty=(dy if region=="dora_indicators" else 0.0)-cy
            offx=self.right_axis.x*tx+self.up_axis.x*ty
            offy=self.right_axis.y*tx+self.up_axis.y*ty
            offz=self.right_axis.z*tx+self.up_axis.z*ty
            px=[];py=[]
            for x,y,z in pts:
                depth=-(z+offz)
                if depth<=.01: return None
                px.append(self.cx+self.fx*(x+offx)/depth)
                py.append(self.cy-self.fx*(y+offy)/depth)
            boxes.append((min(px),min(py),max(px),max(py)))
        return boxes

    def frame_visibility(self, camx,camy):
        crop=self.crops["melds"]
        x,y,w,h=crop
        shift=self.inverse_rotation@Vector((-camx,-camy,0))
        seen={"near":False,"right":False}
        for name,(px,py,pz) in self.frame_samples:
            depth=-(pz+shift.z)
            if depth<=.01: continue
            sx=self.cx+self.fx*(px+shift.x)/depth
            sy=self.cy-self.fx*(py+shift.y)/depth
            if x<=sx<=x+w and y<=sy<=y+h:
                seen[name]=True
        return seen

    def physical_penalty(self, dx,dy):
        # All dora move together; preserve inter-tile arrangements.
        bad=0.0
        moved=[]
        other=[p for reg,p in zip(self.groups,self.polys) if reg!="dora_indicators"]
        for reg,poly in zip(self.groups,self.polys):
            if reg!="dora_indicators":continue
            p2=[(x+dx,y+dy) for x,y in poly]
            moved.append(p2)
            for x,y in p2:
                bad+=max(0.0,abs(x)-self.g.INNER_X/2)**2*5e6
                bad+=max(0.0,abs(y)-self.g.INNER_Y/2)**2*5e6
            if any(self.g.polygons_overlap(p2,p) for p in other):
                bad+=1e6
        return bad

    def evaluate(self, camx,camy,dorax,doray):
        boxes=self.boxes(camx,camy,dorax,doray)
        if boxes is None:return 1e12,{}
        outside=0.0
        foreign=0.0
        n_bad=0
        n_foreign=0
        largest=0.0
        # No bbox clipping allowed; foreign region overlap may be under 10%.
        for reg,b in zip(self.groups,boxes):
            crop=self.crops[reg]
            l,t,r,bt=b
            x,y,w,h=crop
            over=[max(0,x-l),max(0,y-t),max(0,r-(x+w)),max(0,bt-(y+h))]
            largest=max(largest,max(over))
            outside+=sum(v*v for v in over)
            if max(over)>0.01:n_bad+=1
            area=max(1e-6,(r-l)*(bt-t))
            for name,rect in self.crops.items():
                if name==reg:continue
                rate=_intersection_area(b,rect)/area
                ex=max(0,rate-(self.max_foreign_ratio-.0001))
                foreign+=ex*ex*1000
                if rate>=self.max_foreign_ratio:n_foreign+=1
        midpoint=_center(self.tiles,boxes,"completed_hand")
        centering=0 if midpoint is None else (
            ((midpoint[0]-self.cx)/40)**2 + ((midpoint[1]-self.cy)/40)**2
        )
        physical=self.physical_penalty(dorax,doray)
        frame=self.frame_visibility(camx,camy)
        frame_preference=0.003*sum(not flag for flag in frame.values())
        objective=(outside*20+foreign*100+physical+centering*.02
                   +(camx**2+camy**2+dorax**2+doray**2)*.5+frame_preference)
        return objective,{"outside_squared_px":outside,"foreign_penalty":foreign,
                           "bad_own":n_bad,"bad_foreign":n_foreign,
                           "max_outside_px":largest,"completed_center":midpoint,
                           "physical_penalty":physical,
                           "black_frame_visible":frame}

    def solve(self):
        # Broad, bounded exploration of camera translations. Cheap analytical
        # projection means no Blender render calls in the optimization loop.
        starts=[]
        for ix in range(-6,7):
            for iy in range(-5,7):
                cx=ix*.018
                cy=iy*.018
                v,_=self.evaluate(cx,cy,0,0)
                starts.append((v,cx,cy))
        starts.sort()
        best=(1e15,None,None)
        for _,sx,sy in starts[:10]:
            p=[sx,sy,0.0,0.0]
            value,info=self.evaluate(*p)
            for step in (.055,.025,.012,.006,.003,.0015,.0005):
                for _ in range(12):
                    changed=False
                    for axis in (0,1,2,3):
                        for sign in (-1,1):
                            q=p.copy()
                            q[axis]+=sign*step
                            if abs(q[axis])>(.19 if axis<2 else .13):continue
                            v,stats=self.evaluate(*q)
                            if v+1e-10<value:
                                p,value,info=q,v,stats
                                changed=True
                    if not changed:break
            if value<best[0]:best=(value,p,info)
            if info["bad_own"]==0 and info["bad_foreign"]==0 and info["physical_penalty"]==0:
                break
        return best

    def apply(self, p):
        camx,camy,dorax,doray=p
        self.cam.location=self.origin+Vector((camx,camy,0))
        for t in self.tiles:
            if t["region"]!="dora_indicators":continue
            t["root"].location.x+=dorax
            t["root"].location.y+=doray
            t["x"]+=dorax
            t["y"]+=doray
