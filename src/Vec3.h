#pragma once

template <typename T>
class Vec3
{
    T x, y, z;
public:
    explicit Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
    explicit Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
    ~Vec3() {}

    const T& GetX() const { return x; }
    const T& GetY() const { return y; }
    const T& GetZ() const { return z; }

    void SetX(const T& x) { this->x = x; }
    void SetY(const T& y) { this->y = y; }
    void SetZ(const T& z) { this->z = z; }

    T LengthSquare() const {
        return x * x + y * y + z * z;
    }

    T Length() const {
        return sqrt(LengthSquare());
    }

    Vec3& Normalize()
    {
        T lengthSquare = LengthSquare();

        if (lengthSquare > 0.0f) {
            T inverseLength = 1.0f / sqrt(lengthSquare);

            x *= inverseLength;
            y *= inverseLength;
            z *= inverseLength;
        }

        return *this;
    }

    T dot(const Vec3<T>& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    Vec3<T> operator - (const Vec3<T>& v) const {
        return Vec3<T>(x - v.x, y - v.y, z - v.z);
    }

    Vec3<T> operator + (const Vec3<T>& v) const {
        return Vec3<T>(x + v.x, y + v.y, z + v.z);
    }

    Vec3<T> operator * (const T &f) const {
        return Vec3<T>(x * f, y * f, z * f);
    }

    Vec3<T> operator * (const Vec3<T> &f) const {
        return Vec3<T>(x * f.x, y * f.y, z * f.z);
    }

    Vec3<T> operator - () const {
        return Vec3<T>(-x, -y, -z);
    }


    Vec3<T>& operator += (const Vec3<T> &v) {
        x += v.x, y += v.y, z += v.z;
        return *this;
    }
};

template<typename T>
Vec3<T> operator * (const T& f, const Vec3<T>&v) {
    return Vec3<T>(v.GetX() * f, v.GetY() * f, v.GetZ() * f);
}

typedef Vec3<float> Vec3f;