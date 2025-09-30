from typing import Any, Union, overload

import torch

HANDLED_FUNCTIONS = {}
FUNCTIONS_RETURNING_QUATERNION = {
    torch.add,
    torch.sub,
    torch.mul,
    torch.pow,
    torch.exp,
    torch.log,
    torch.real,
    torch.imag,
}


def CHECK_OPERAND_SHAPE(other: Any, scalar_allowed: bool = True):
    if torch.is_tensor(other) and other.shape[-1] not in (1, 4):
        raise ValueError(
            "The last dimension must be of size 4 to represent a quaternion (WXYZ) or size 1 to represent a real scalar."
        )
    if torch.is_tensor(other) and other.dtype in [torch.complex64, torch.complex128]:
        raise TypeError("Cannot operate between quaternion and complex tensors.")
    if not scalar_allowed and isinstance(other, (int, float, complex)):
        raise TypeError("Operand must not be a scalar.")


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class Quaternion(torch.Tensor):
    r"""A ``torch.Tensor`` subclass representing quaternions, defined as

    .. math::
        q = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}


    Quaternions are represented as tensors of shape :math:`(..., 4)`, where the last dimension
    corresponds to the components :math:`(W, X, Y, Z)`.

    Args:
        w: The real part of the quaternion.
        x: The first imaginary part of the quaternion.
        y: The second imaginary part of the quaternion.
        z: The third imaginary part of the quaternion.

    or

    Args:
        data: A tensor of shape :math:`(..., 4)` representing the quaternion components
            in the order :math:`(W, X, Y, Z)`.

    """

    def __new__(cls, *args, **kwargs):
        if "data" in kwargs or (len(args) == 1 and isinstance(args[0], torch.Tensor)):
            return super().__new__(cls, *args, **kwargs)
        if len(args) == 4:
            tensors = tuple(torch.as_tensor(arg) for arg in args)
            try:
                data = torch.stack(tensors, dim=-1)
            except RuntimeError as e:
                raise ValueError("All input tensors must have the same shape.") from e
            return super().__new__(cls, data)
        if all(_ in kwargs for _ in "wxyz"):
            return super().__new__(
                cls,
                torch.stack(
                    [kwargs["w"], kwargs["x"], kwargs["y"], kwargs["z"]], dim=-1
                ),
            )
        raise ValueError("Invalid arguments for Quaternion initialization.")

    @overload
    def __init__(self, data: torch.Tensor, *args, **kwargs): ...

    @overload
    def __init__(
        self,
        w: Union[float, torch.Tensor],
        x: Union[float, torch.Tensor],
        y: Union[float, torch.Tensor],
        z: Union[float, torch.Tensor],
        **kwargs,
    ): ...

    def __init__(self, *args, **kwargs):
        super().__init__()
        if self.shape[-1] != 4:
            raise ValueError(
                "The last dimension must be of size 4 to represent a quaternion (WXYZ)."
            )

    def to_wxyz(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.as_subclass(torch.Tensor).unbind(-1)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

        result = super().__torch_function__(func, types, args, kwargs)
        if (
            isinstance(result, Quaternion)
            and func not in FUNCTIONS_RETURNING_QUATERNION
        ):
            result = result.as_subclass(torch.Tensor)
        return result

    @implements(torch.add)
    def add(self, other: Union[float, "Quaternion"]) -> "Quaternion":
        """Quaternion addition (element-wise)."""
        CHECK_OPERAND_SHAPE(other, scalar_allowed=False)

        return Quaternion(super().add(other))

    @implements(torch.mul)
    def mul(self, other: Union[float, "Quaternion"]) -> "Quaternion":
        """Non-commutative quaternion multiplication."""
        CHECK_OPERAND_SHAPE(other, scalar_allowed=True)

        if isinstance(other, (int, float)):
            return Quaternion(super().mul(other))

        w1, x1, y1, z1 = self.to_wxyz()
        if other.shape[-1] == 4:
            w2, x2, y2, z2 = Quaternion(other).to_wxyz()
        else:  # scalar
            w2 = other[..., 0]
            x2 = 0
            y2 = 0
            z2 = 0

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Quaternion(torch.stack([w, x, y, z], dim=-1))

    @implements(torch.sub)
    def sub(self, other: Union[float, "Quaternion"]) -> "Quaternion":
        """Quaternion subtraction (element-wise)."""
        CHECK_OPERAND_SHAPE(other, scalar_allowed=False)

        return Quaternion(super().sub(other))

    @implements(torch.abs)
    def abs(self):
        r"""Quaternion norm, defined as

        .. math::
            \|q\| = \sqrt{w^2 + x^2 + y^2 + z^2}

        Returns:
            A tensor of shape :math:`(...)` representing the norm of the quaternion(s).
        """
        w, x, y, z = self.to_wxyz()
        return torch.sqrt(w**2 + x**2 + y**2 + z**2)

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __mul__(self, other):
        return self.mul(other)

    def __pow__(self, other):
        return self.pow(other)

    def __truediv__(self, other):
        if isinstance(other, Quaternion):
            return self.mul(other.inverse())
        return self.div(other)

    def conjugate(self):
        r"""Quaternion conjugate, defined as:math:`q^* = w - x\mathbf{i} - y\mathbf{j} - z\mathbf{k}`."""

        w, x, y, z = self.to_wxyz()
        return Quaternion(torch.stack([w, -x, -y, -z], dim=-1))

    def inverse(self):
        r"""Quaternion inverse, defined as :math:`q^{-1} = \frac{q^*}{\|q\|^2}`."""
        norm_sq = self.abs() ** 2
        conj = self.conjugate()
        return Quaternion(conj / norm_sq.unsqueeze(-1))

    def normalize(self):
        r"""Returns a normalized quaternion, defined as :math:`\frac{q}{\|q\|}`."""
        norm = self.abs()
        return Quaternion(self / norm.unsqueeze(-1))

    def to_rotation_matrix(self) -> torch.Tensor:
        r"""Convert the quaternion to a 3x3 rotation matrix.

        Returns:
            A tensor of shape :math:`(..., 3, 3)` representing the rotation matrix.
        """
        w, x, y, z = self.to_wxyz()
        leading_dims = self.shape[:-1]
        rotation_matrix = torch.empty(
            *leading_dims, 3, 3, device=self.device, dtype=self.dtype
        )

        rotation_matrix[..., 0, 0] = 1 - 2 * (y**2 + z**2)
        rotation_matrix[..., 0, 1] = 2 * (x * y - z * w)
        rotation_matrix[..., 0, 2] = 2 * (x * z + y * w)

        rotation_matrix[..., 1, 0] = 2 * (x * y + z * w)
        rotation_matrix[..., 1, 1] = 1 - 2 * (x**2 + z**2)
        rotation_matrix[..., 1, 2] = 2 * (y * z - x * w)

        rotation_matrix[..., 2, 0] = 2 * (x * z - y * w)
        rotation_matrix[..., 2, 1] = 2 * (y * z + x * w)
        rotation_matrix[..., 2, 2] = 1 - 2 * (x**2 + y**2)

        return rotation_matrix

    @staticmethod
    def from_rotation_matrix(R: torch.Tensor) -> "Quaternion":
        r"""Create a quaternion from a 3x3 rotation matrix.

        Args:
            R: A tensor of shape :math:`(..., 3, 3)` representing the rotation matrix(or matrices).
        Returns:
            An equivalent quaternion.
        """
        if R.shape[-2:] != (3, 3):
            raise ValueError("Input rotation matrix must have shape (..., 3, 3)")
        B = R.shape[:-2]
        R = R.reshape(-1, 3, 3)

        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        w = torch.sqrt(1.0 + trace) / 2.0
        x = (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * w)
        y = (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * w)
        z = (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * w)

        q = torch.stack([w, x, y, z], dim=-1)
        q = q.reshape(*B, 4)
        return Quaternion(q)

    @staticmethod
    def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> "Quaternion":
        r"""Create a quaternion from an axis-angle representation.
        Args:
            axis: A tensor of shape :math:`(..., 3)` representing the rotation axis
            angle: A tensor of shape :math:`(...)` representing the rotation angle in radians

        Returns:
            An equivalent quaternion.
        """
        if axis.shape[-1] != 3:
            raise ValueError("Axis must have shape (..., 3)")
        if axis.dim() != angle.dim() + 1:
            raise ValueError(
                "Axis (..., 3) and angle (...) must have the same number of leading dimensions"
            )
        if axis.shape[:-1] != angle.shape:
            raise ValueError("Axis and angle must have compatible shapes")

        half_angle = angle / 2.0
        sin_half_angle = torch.sin(half_angle)
        cos_half_angle = torch.cos(half_angle)

        axis = axis / torch.norm(axis, dim=-1, keepdim=True)

        w = cos_half_angle
        x = axis[..., 0] * sin_half_angle
        y = axis[..., 1] * sin_half_angle
        z = axis[..., 2] * sin_half_angle

        q = torch.stack([w, x, y, z], dim=-1)
        return Quaternion(q)

    def to_axis_angle(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Convert the quaternion to an axis-angle representation.

        Returns:
            A tuple containing:
            - A tensor of shape :math:`(..., 3)` representing the rotation axis.
            - A tensor of shape :math:`(...)` representing the rotation angle in radians.
        """
        w, x, y, z = self.to_wxyz()
        angle = 2 * torch.acos(w)
        s = torch.sqrt(1 - w**2)
        s = torch.where(s < 1e-8, torch.tensor(1e-8, device=s.device, dtype=s.dtype), s)
        axis = torch.stack([x / s, y / s, z / s], dim=-1)
        return axis, angle

    def rotate_vector(self, v: torch.Tensor):
        """Rotate a 3D vector or a batch of 3D vectors using this quaternion.

        Args:
            v: A tensor of shape :math:`(..., 3)` representing the 3D vector(s) to be rotated.

        Returns:
            A tensor of shape :math:`(..., 3)` representing the rotated vector(s).
        """

        if v.shape[-1] != 3:
            raise ValueError("Input vector must have shape (..., 3)")

        v_quat = Quaternion(
            torch.zeros_like(v[..., 0]), v[..., 0], v[..., 1], v[..., 2]
        )

        rotated_v_quat = self * v_quat * self.conjugate()
        return torch.stack(
            [rotated_v_quat[..., 1], rotated_v_quat[..., 2], rotated_v_quat[..., 3]],
            dim=-1,
        )

    def slerp(self, other: "Quaternion", t: Union[float, torch.Tensor]):
        """Performs spherical linear interpolation (slerp) between this quaternion and another quaternion.

        Args:
            other: The target quaternion to interpolate towards.
            t: The interpolation factor, where 0.0 corresponds to this quaternion and 1.0 corresponds to the other quaternion.

        Returns:
            The interpolated quaternion.
        """
        CHECK_OPERAND_SHAPE(other, scalar_allowed=False)
        if self.shape != other.shape:
            raise ValueError("Quaternions must have the same shape for slerp.")
        if isinstance(t, (int, float)):
            t = torch.tensor(t, device=self.device, dtype=self.dtype)

        return self * (self.inverse() * other) ** t

    @implements(torch.log)
    def log(self) -> "Quaternion":
        r"""Quaternion logarithm of :math:`q`, defined as

        .. math::
            \log q=\log {\|q\|} +{\frac {\mathbf {v} }{\|\mathbf {v} \|}}\arccos {\frac {w}{\|q\|}}


        where :math:`q = w + \mathbf{v}` with :math:`w`  the real part and :math:`\mathbf{v}` the vector part of the quaternion.
        """
        w, x, y, z = self.to_wxyz()
        v_norm = torch.sqrt(x**2 + y**2 + z**2)
        v_norm = torch.where(
            v_norm < 1e-8,
            torch.tensor(1e-8, device=v_norm.device, dtype=v_norm.dtype),
            v_norm,
        )
        theta = torch.atan2(
            v_norm, w
        )  # note it's v_norm, not q_norm here. It's equivalent and atan2 is more stable
        coeff = theta / v_norm
        return Quaternion(
            torch.stack(
                [torch.log(self.abs()), x * coeff, y * coeff, z * coeff], dim=-1
            )
        )

    @implements(torch.exp)
    def exp(self) -> "Quaternion":
        r"""Quaternion exponential of :math:`q`, defined as

        .. math::
            e^q=e^{w}\left(\cos \|\mathbf {v} \|+{\frac {\mathbf {v} }{\|\mathbf {v} \|}}\sin \|\mathbf {v} \|\right)

        where :math:`q = w + \mathbf{v}` with :math:`{w}` the real part and :math:`\mathbf{v}` the vector part of the quaternion.

        """
        w, x, y, z = self.to_wxyz()
        v_norm = torch.sqrt(x**2 + y**2 + z**2)
        exp_w = torch.exp(w)
        cos_v_norm = torch.cos(v_norm)
        coeff = torch.sinc(v_norm / torch.pi)  # sin(x)/x, more stable for small x
        return Quaternion(
            torch.stack(
                [
                    exp_w * cos_v_norm,
                    exp_w * x * coeff,
                    exp_w * y * coeff,
                    exp_w * z * coeff,
                ],
                dim=-1,
            )
        )

    @implements(torch.pow)
    def pow(self, exponent: Union[float, torch.Tensor, "Quaternion"]) -> "Quaternion":
        r"""Quaternion power, defined as

        .. math::
            q^t = \exp(t \log q)

        where :math:`t` is the exponent.
        """
        if isinstance(exponent, (int, float)):
            exponent = torch.tensor(exponent, device=self.device, dtype=self.dtype)
        if exponent.dim() == 0:
            exponent = exponent.unsqueeze(0)

        log_q = self.log()
        scaled_log_q = Quaternion(log_q * exponent)
        return scaled_log_q.exp()

    @implements(torch.real)
    def real(self) -> "Quaternion":
        """Real part of quaternion, i.e., :math:`w`

        Returns:
            A real quaternion
        """
        w, _, _, _ = self.to_wxyz()
        zero = torch.zeros_like(w)
        return Quaternion(w, zero, zero, zero)

    @implements(torch.imag)
    def imag(self) -> "Quaternion":
        """Imaginary part of quaternion, i.e., :math:`xi + yj + zk`

        Returns:
            A pure imaginary quaternion
        """
        _, x, y, z = self.to_wxyz()
        zero = torch.zeros_like(x)
        return Quaternion(zero, x, y, z)
