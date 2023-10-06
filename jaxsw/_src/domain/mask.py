import typing as tp
from jaxtyping import Array
import jax.numpy as jnp
from jaxsw._src.utils.pooling import avg_pool_2d

class Mask(tp.NamedTuple):
    q: Array
    u: Array
    v: Array
    psi: Array
    not_q: Array
    not_psi: Array
    not_u: Array
    not_v: Array
    psi_irrbound_xids: Array
    psi_irrbound_yids: Array
    q_distbound1: Array
    u_distbound1: Array
    v_distbound1: Array
    u_distbound2plus: Array
    v_distbound2plus: Array
    u_distbound2: Array
    v_distbound2: Array
    u_distbound3plus: Array
    v_distbound3plus: Array
    
    @classmethod
    def init_mask(cls, mask: Array, variable: str="q"):
        
        mtype = mask.dtype
        
        if variable == "q":
            q, u, v, psi = init_mask_from_q(mask)
        elif variable == "psi":
            q, u, v, psi = init_mask_from_psi(mask)
        else:
            raise ValueError(f"Unrecognized variable: {variable}")
        not_q = jnp.logical_not(q.astype(bool))
        not_u = jnp.logical_not(u.astype(bool))
        not_v = jnp.logical_not(v.astype(bool))
        not_psi = jnp.logical_not(psi.astype(bool))
        
        psi_irrbound_xids = jnp.logical_and(
            not_psi[1:-1,1:-1],
            avg_pool_2d(psi, kernel_size=(3,3), strides=(1,1), padding=(0,0)) > 1/18
        )
        psi_irrbound_xids = jnp.where(psi_irrbound_xids)

        q_distbound1 = jnp.logical_and(
            avg_pool_2d(q, kernel_size=(3,3), strides=(1,1), padding=(1,1)) < 17/18,
            q
        )
        u_distbound1 = jnp.logical_and(
            avg_pool_2d(u, kernel_size=(3,1), strides=(1,1), padding=(1,0)) < 5/6,
            u
        )
        v_distbound1 = jnp.logical_and(
            avg_pool_2d(v, kernel_size=(1,3), strides=(1,1), padding=(0,1)) < 5/6,
            v
        )
        
        u_distbound2plus = jnp.logical_and(
            jnp.logical_not(u_distbound1),
            u
        )
        v_distbound2plus = jnp.logical_and(
            jnp.logical_not(v_distbound1),
            v
        )
        
        u_distbound2 = jnp.logical_and(
            avg_pool_2d(u, kernel_size=(5,1), strides=(1,1), padding=(2,0)) < 9/10,
            u_distbound2plus
        )
        v_distbound2 = jnp.logical_and(
            avg_pool_2d(v, kernel_size=(1,5), strides=(1,1), padding=(0,2)) < 9/10,
            v_distbound2plus
        )

        u_distbound3plus = jnp.logical_and(
            jnp.logical_not(u_distbound2), 
            u_distbound2plus
        )
        v_distbound3plus = jnp.logical_and(
            jnp.logical_not(v_distbound2), 
            v_distbound2plus
        )



        return cls(
            q=q.astype(mtype), 
            u=u.astype(mtype), 
            v=v.astype(mtype), 
            psi=psi.astype(mtype),
            not_q=not_q, not_u=not_u.astype(mtype),
            not_v=not_v, not_psi=not_psi.astype(mtype),
            psi_irrbound_xids=psi_irrbound_xids[0].astype(mtype),
            psi_irrbound_yids=psi_irrbound_xids[1].astype(mtype),
            q_distbound1=q_distbound1.astype(mtype),
            u_distbound1=u_distbound1.astype(mtype),
            v_distbound1=v_distbound1.astype(mtype),
            u_distbound2plus=u_distbound2plus.astype(mtype),
            v_distbound2plus=v_distbound2plus.astype(mtype),
            u_distbound2=u_distbound2.astype(mtype),
            v_distbound2=v_distbound2.astype(mtype),
            u_distbound3plus=u_distbound3plus.astype(mtype),
            v_distbound3plus=v_distbound3plus.astype(mtype),
        )
        

    
    
def init_mask_from_q(mask: Array):
    
    q = jnp.copy(mask)
    u = avg_pool_2d(q, kernel_size=(2,1), strides=(1,1), padding=(1,0)) > 3/4
    v = avg_pool_2d(q, kernel_size=(1,2), strides=(1,1), padding=(0,1)) > 3/4
    psi = avg_pool_2d(q, kernel_size=(2,2), strides=(1,1), padding=(1,1)) > 7/8
    return q, u, v, psi


def init_mask_from_psi(mask: Array):
    
    psi = jnp.copy(mask)
    q = avg_pool_2d(psi, kernel_size=(2,2), strides=(1,1), padding=(0,0)) > 0.0
    u = avg_pool_2d(q, kernel_size=(2,1), strides=(1,1), padding=(1,0)) > 3/4
    v = avg_pool_2d(q, kernel_size=(1,2), strides=(1,1), padding=(0,1)) > 3/4
    
    return q, u, v, psi