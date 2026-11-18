/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/FaissAssert.h>
#include <faiss/python/python_callbacks.h>

namespace {

struct PyThreadLock {
    PyGILState_STATE gstate;

    PyThreadLock() {
        gstate = PyGILState_Ensure();
    }

    ~PyThreadLock() {
        PyGILState_Release(gstate);
    }
};

} // namespace

/***********************************************************
 * Callbacks for IO reader and writer
 ***********************************************************/

PyCallbackIOWriter::PyCallbackIOWriter(PyObject* callback, size_t bs)
        : callback(callback), bs(bs) {
    PyThreadLock gil;
    Py_INCREF(callback);
    name = "PyCallbackIOWriter";
}

size_t PyCallbackIOWriter::operator()(
        const void* ptrv,
        size_t size,
        size_t nitems) {
    size_t ws = size * nitems;
    const char* ptr = (const char*)ptrv;
    PyThreadLock gil;
    while (ws > 0) {
        size_t wi = ws > bs ? bs : ws;
        PyObject* result = PyObject_CallFunction(
                callback, "(N)", PyBytes_FromStringAndSize(ptr, wi));
        if (result == nullptr) {
            FAISS_THROW_MSG("py err");
        }
        // TODO check nb of bytes written
        ptr += wi;
        ws -= wi;
        Py_DECREF(result);
    }
    return nitems;
}

PyCallbackIOWriter::~PyCallbackIOWriter() {
    PyThreadLock gil;
    Py_DECREF(callback);
}

PyCallbackIOReader::PyCallbackIOReader(PyObject* callback, size_t bs)
        : callback(callback), bs(bs) {
    PyThreadLock gil;
    Py_INCREF(callback);
    name = "PyCallbackIOReader";
}

size_t PyCallbackIOReader::operator()(void* ptrv, size_t size, size_t nitems) {
    size_t rs = size * nitems;
    size_t nb = 0;
    char* ptr = (char*)ptrv;
    PyThreadLock gil;
    while (rs > 0) {
        size_t ri = rs > bs ? bs : rs;
        PyObject* result = PyObject_CallFunction(callback, "(n)", ri);
        if (result == nullptr) {
            FAISS_THROW_MSG("propagate py error");
        }
        if (!PyBytes_Check(result)) {
            Py_DECREF(result);
            FAISS_THROW_MSG("read callback did not return a bytes object");
        }
        size_t sz = PyBytes_Size(result);
        if (sz == 0) {
            Py_DECREF(result);
            break;
        }
        nb += sz;
        if (sz > rs) {
            Py_DECREF(result);
            FAISS_THROW_FMT(
                    "read callback returned %zd bytes (asked %zd)", sz, rs);
        }
        memcpy(ptr, PyBytes_AsString(result), sz);
        Py_DECREF(result);
        ptr += sz;
        rs -= sz;
    }
    return nb / size;
}

PyCallbackIOReader::~PyCallbackIOReader() {
    PyThreadLock gil;
    Py_DECREF(callback);
}

/***********************************************************
 * Callbacks for IDSelector
 ***********************************************************/

PyCallbackIDSelector::PyCallbackIDSelector(PyObject* callback)
        : callback(callback) {
    PyThreadLock gil;
    Py_INCREF(callback);
}

bool PyCallbackIDSelector::is_member(faiss::idx_t id) const {
    FAISS_THROW_IF_NOT((id >> 32) == 0);
    PyThreadLock gil;
    PyObject* result = PyObject_CallFunction(callback, "(n)", int(id));
    if (result == nullptr) {
        FAISS_THROW_MSG("propagate py error");
    }
    bool b = PyObject_IsTrue(result);
    Py_DECREF(result);
    return b;
}

PyCallbackIDSelector::~PyCallbackIDSelector() {
    PyThreadLock gil;
    Py_DECREF(callback);
}

/***********************************************************
 * Callbacks for intermediate search results
 ***********************************************************/

PyCallbackIntermediateResult::PyCallbackIntermediateResult(PyObject* callback)
        : callback(callback) {
    PyThreadLock gil;
    Py_INCREF(callback);
}

void PyCallbackIntermediateResult::on_result(
        faiss::idx_t query_idx,
        faiss::idx_t k,
        const float* distances,
        const faiss::idx_t* labels,
        size_t probes_completed,
        size_t total_probes) {
    PyThreadLock gil;
    PyObject* py_distances = PyList_New(k);
    PyObject* py_labels = PyList_New(k);
    for (faiss::idx_t i = 0; i < k; i++) {
        PyList_SET_ITEM(py_distances, i, PyFloat_FromDouble(distances[i]));
        PyList_SET_ITEM(py_labels, i, PyLong_FromLongLong(labels[i]));
    }
    PyObject* result = PyObject_CallFunction(
            callback,
            "(LOOnn)",
            (long long)query_idx,
            py_distances,
            py_labels,
            (Py_ssize_t)probes_completed,
            (Py_ssize_t)total_probes);
    if (result == nullptr) {
        Py_DECREF(py_distances);
        Py_DECREF(py_labels);
        FAISS_THROW_MSG("intermediate result callback raised an exception");
    }
    Py_DECREF(result);
    Py_DECREF(py_distances);
    Py_DECREF(py_labels);
}

PyCallbackIntermediateResult::~PyCallbackIntermediateResult() {
    PyThreadLock gil;
    Py_DECREF(callback);
}

/***********************************************************
 * Callbacks for IVF index sharding
 ***********************************************************/

PyCallbackShardingFunction::PyCallbackShardingFunction(PyObject* callback)
        : callback(callback) {
    PyThreadLock gil;
    Py_INCREF(callback);
}

int64_t PyCallbackShardingFunction::operator()(int64_t i, int64_t shard_count) {
    PyThreadLock gil;
    PyObject* shard_id = PyObject_CallFunction(callback, "LL", i, shard_count);
    if (shard_id == nullptr) {
        FAISS_THROW_MSG("propagate py error");
    }
    return PyLong_AsLongLong(shard_id);
}

PyCallbackShardingFunction::~PyCallbackShardingFunction() {
    PyThreadLock gil;
    Py_DECREF(callback);
}
