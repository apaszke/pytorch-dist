#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.cpp"
#else

PyObject *THPStorageClass = NULL;

PyObject * THPStorage_(New)(THStorage *ptr)
{
  THPObjectPtr args = PyTuple_New(0);
  THPObjectPtr kwargs;
  THPUtils_assert(args, "Could not create a new storage object - failed to"
          "allocate argument tuple");
  if (ptr) {
    kwargs = Py_BuildValue("{s:N}", "cdata", PyLong_FromVoidPtr(ptr));
    THPUtils_assert(kwargs, "Could not create a new storage object - "
          "failed to allocate keyword argument dictionary");
  }
  PyObject *result = PyObject_Call(THPStorageClass, args, kwargs);
  return result;
}

// TODO: THD
#ifndef THD_GENERIC_FILE
PyObject * THPStorage_(newWeakObject)(THStorage *storage) {
#ifdef THC_GENERIC_FILE
  return PyErr_Format(PyExc_TypeError, "newWeakObject not supported on CUDA storages");
#else
  if (storage->allocator == &THStorageWeakRefAllocator) {
    auto allocator_obj = ((StorageWeakRefAllocator*)storage->allocatorContext);
    Py_INCREF(allocator_obj->object.get());
    return allocator_obj->object.get();
  }
  std::unique_ptr<StorageWeakRefAllocator> new_ctx(new StorageWeakRefAllocator(
        nullptr, storage->allocator, storage->allocatorContext));
  PyObject *weak_result = THPStorage_(New)(storage);
  if (!weak_result)
    return NULL;
  Py_INCREF(weak_result); // THPObjectPtr steals the reference
  new_ctx->object = weak_result;
  storage->allocatorContext = (void*)new_ctx.release();
  storage->allocator = &THStorageWeakRefAllocator;
  return weak_result;
#endif
}
#endif

static void THPStorage_(dealloc)(THPStorage* self)
{
  if (self->cdata)
    THStorage_(free)(LIBRARY_STATE self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// TODO: THD
#ifndef THD_GENERIC_FILE
static THStorage* THPStorage_(newWithAllocator)(long size, THAllocator* allocator)
{
#ifdef THC_GENERIC_FILE
  THPUtils_setError(THPStorageStr " does not support custom allocators");
  return NULL;
#else
  return THStorage_(newWithAllocator)(LIBRARY_STATE size, allocator, NULL);
#endif
}
#endif

static PyObject * THPStorage_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  Py_ssize_t num_args = args ? PyTuple_Size(args) : 0;

  THPStoragePtr self = (THPStorage *)type->tp_alloc(type, 0);
  THPUtils_assert(self, "failed to allocate a " THPStorageStr " object");
// TODO: THD
#ifndef THD_GENERIC_FILE
  THAllocator* allocator = NULL;
#endif

  // Internally we allow constructing with a keywoard only argument cdata
  if (kwargs != NULL) {
    PyObject *allocator_ptr = PyDict_GetItemString(kwargs, "allocator");
// TODO: THD
#ifndef THD_GENERIC_FILE
    if (allocator_ptr) {
      THPUtils_assert(THPUtils_checkLong(allocator_ptr), "invalid allocator");
      allocator = (THAllocator*) PyLong_AsVoidPtr(allocator_ptr);
      PyDict_DelItemString(kwargs, "allocator");
    }
#else
    if (allocator_ptr) {
      THPUtils_setError("distributed storages don't support custom allocators");
      return NULL;
    }
#endif

    Py_ssize_t num_kwargs = PyDict_Size(kwargs);
    if (num_args == 0) {
      PyObject *cdata_ptr = PyDict_GetItemString(kwargs, "cdata");
      if (num_kwargs == 1 && cdata_ptr && THPUtils_checkLong(cdata_ptr)) {
        THStorage *ptr = (THStorage*)PyLong_AsVoidPtr(cdata_ptr);
        self->cdata = ptr;
        return (PyObject*)self.release();
      }
    }
    THPUtils_assert(num_kwargs == 0, THPStorageStr "(): invalid keyword arguments");
  }

  // torch.Storage()
  if (num_args == 0) {
// TODO: THD
#ifndef THD_GENERIC_FILE
    if (allocator) {
      self->cdata = THPStorage_(newWithAllocator)(0, allocator);
    } else {
#else
    if (true) {
#endif
      self->cdata = THStorage_(new)(LIBRARY_STATE_NOARGS);
    }
    return (PyObject*)self.release();
  }

  PyObject *first_arg = PyTuple_GET_ITEM(args, 0);

  // torch.Storage(size)
  if (num_args == 1 && THPUtils_checkLong(first_arg)) {
    long size = THPUtils_unpackLong(first_arg);
// TODO: THD
#ifndef THD_GENERIC_FILE
    if (allocator) {
      self->cdata = THPStorage_(newWithAllocator)(size, allocator);
    } else {
#else
    if (true) {
#endif
      self->cdata = THStorage_(newWithSize)(LIBRARY_STATE size);
    }
    return (PyObject*)self.release();
  }

  // torch.Storage(view_source, [offset, [size]])
// TODO: THD
#ifndef THD_GENERIC_FILE
  if (num_args < 4 && THPStorage_(Check)(first_arg)) {
    THPStorage *storage_arg = (THPStorage *)first_arg;
    long numel = storage_arg->cdata->size;
    long offset = 0;

    if (num_args >= 2) {
      PyObject *second_arg = PyTuple_GET_ITEM(args, 1);
      if (!THPUtils_checkLong(second_arg))
        goto invalid_arguments;
      offset = THPUtils_unpackLong(second_arg);
    }

    long size = numel - offset;
    if (num_args >= 3) {
      PyObject *third_arg = PyTuple_GET_ITEM(args, 2);
      if (!THPUtils_checkLong(third_arg))
        goto invalid_arguments;
      size = THPUtils_unpackLong(third_arg);
    }

    THPUtils_assert(offset >= 0 && offset <= numel, "specified an offset of "
        "%ld, but the viewed storage has only %ld element(s)", offset, numel);
    THPUtils_assert(size >= 1 && size <= numel - offset, "specified a size of "
        "%d, but the viewed storage has only %ld element(s) after offset %ld",
        size, numel - offset, offset);

    real *data_ptr = storage_arg->cdata->data + offset;
    THStoragePtr storage = THStorage_(newWithData)(LIBRARY_STATE data_ptr, size);
    storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_VIEW;
    storage->view = storage_arg->cdata;
    THStorage_(retain)(LIBRARY_STATE storage_arg->cdata);
    self->cdata = storage.release();
    return (PyObject*)self.release();
  }
#endif

// TODO: THD
#ifndef THD_GENERIC_FILE
  // torch.Storage(sequence)
  if (num_args == 1 && PySequence_Check(first_arg)) {
    Py_ssize_t length = PySequence_Length(first_arg);
    THPUtils_assert(length >= 0, "couldn't obtain the length of %s",
        THPUtils_typename(first_arg));
    self->cdata = THStorage_(newWithSize)(LIBRARY_STATE length);
    THPObjectPtr item;
    try {
      for (Py_ssize_t i = 0; i < length; i++) {
        item = PySequence_GetItem(first_arg, i);
        real value = THPUtils_(unpackReal)(item.get());
#ifndef THC_GENERIC_FILE
        self->cdata->data[i] = value;
#else
        // TODO: this might be slow - consider batched updates?
        THCStorage_(set)(LIBRARY_STATE self->cdata, i, value);
#endif
      }
    } catch (std::runtime_error &e) {
      THPUtils_setError("tried to construct a storage from a sequence (%s), "
          "but one of the items was of type %s instead of %s",
          THPUtils_typename(first_arg),
          THPUtils_typename(item.get()),
          THPUtils_typeTraits<real>::python_type_str);
      return NULL;
    }
    return (PyObject*)self.release();
  }
#endif

// TODO: THD
#ifndef THD_GENERIC_FILE
invalid_arguments:
#endif
  THPUtils_invalidArguments(args, THPStorageStr " constructor", 6,
          "no arguments",
          "(int size)",
          "(Sequence data)",
          "(" THPStorageStr " view_source)",
          "(" THPStorageStr " view_source, int offset)",
          "(" THPStorageStr " view_source, int offset, int size)");
  return NULL;
  END_HANDLE_TH_ERRORS
}

static Py_ssize_t THPStorage_(length)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return THStorage_(size)(LIBRARY_STATE self->cdata);
  END_HANDLE_TH_ERRORS_RET(-1)
}

// TODO: THD
#ifndef THD_GENERIC_FILE
static PyObject * THPStorage_(get)(THPStorage *self, PyObject *index)
{
  HANDLE_TH_ERRORS
  /* Integer index */
  if (THPUtils_checkLong(index)) {
    long nindex = THPUtils_unpackLong(index);
    if (nindex < 0)
      nindex += THStorage_(size)(LIBRARY_STATE self->cdata);
    real value = THStorage_(get)(LIBRARY_STATE self->cdata, nindex);
    return THPUtils_(newReal)(value);
  /* Slice index */
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop, slicelength;
    long len = THStorage_(size)(LIBRARY_STATE self->cdata);
    if (!THPUtils_parseSlice(index, len, &start, &stop, &slicelength))
      return NULL;

    real *data = THStorage_(data)(LIBRARY_STATE self->cdata);
    THStoragePtr new_storage = THStorage_(newWithData)(LIBRARY_STATE data + start, slicelength);
    new_storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_VIEW;
    new_storage->view = self->cdata;
    THStorage_(retain)(LIBRARY_STATE self->cdata);

    PyObject *_ret = THPStorage_(New)(new_storage);
    new_storage.release();
    return _ret;
  }
  THPUtils_setError("can't index a " THPStorageStr " with %s",
      THPUtils_typename(index));
  return NULL;
  END_HANDLE_TH_ERRORS
}

static int THPStorage_(set)(THPStorage *self, PyObject *index, PyObject *value)
{
  HANDLE_TH_ERRORS
  if (!THPUtils_(checkReal)(value)) {
    THPUtils_setError("can only set storage content with a %s, but got "
        "%s instead", THPUtils_typeTraits<real>::python_type_str,
        THPUtils_typename(value));
    return -1;
  }

  real rvalue = THPUtils_(unpackReal)(value);
  if (THPUtils_checkLong(index)) {
    long nindex = THPUtils_unpackLong(index);
    THStorage_(set)(LIBRARY_STATE self->cdata, nindex, rvalue);
    return 0;
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop;
    long len = THStorage_(size)(LIBRARY_STATE self->cdata);
    if (!THPUtils_parseSlice(index, len, &start, &stop, NULL))
      return -1;
    // TODO: check the bounds only once
    // TODO: fill?
    for (;start < stop; start++)
      THStorage_(set)(LIBRARY_STATE self->cdata, start, rvalue);
    return 0;
  }
  THPUtils_setError("can't index a " THPStorageStr " with %s",
      THPUtils_typename(index));
  return -1;
  END_HANDLE_TH_ERRORS_RET(-1)
}
#endif

static PyMappingMethods THPStorage_(mappingmethods) = {
  (lenfunc)THPStorage_(length),
#ifndef THD_GENERIC_FILE
  (binaryfunc)THPStorage_(get),
  (objobjargproc)THPStorage_(set)
#endif
};

// TODO: implement equality
PyTypeObject THPStorageType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C." THPStorageBaseStr,         /* tp_name */
  sizeof(THPStorage),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPStorage_(dealloc),      /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  &THPStorage_(mappingmethods),          /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,   /* will be assigned in init */    /* tp_methods */
  0,   /* will be assigned in init */    /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPStorage_(pynew),                    /* tp_new */
};

static struct PyMemberDef THPStorage_(members)[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPStorage, cdata), READONLY, NULL},
  {NULL}
};

// TODO: THD
#ifndef THD_GENERIC_FILE
#include "StorageMethods.cpp"
#endif

bool THPStorage_(init)(PyObject *module)
{
// TODO: THD
#ifndef THD_GENERIC_FILE
  THPStorageType.tp_methods = THPStorage_(methods);
#endif
  THPStorageType.tp_members = THPStorage_(members);
  if (PyType_Ready(&THPStorageType) < 0)
    return false;
  Py_INCREF(&THPStorageType);
  PyModule_AddObject(module, THPStorageBaseStr, (PyObject *)&THPStorageType);
  return true;
}

#endif
